"""This file handles variable types, function signatures and struct layouts
based on a C AST. Based on the pycparser library."""

import copy
from dataclasses import dataclass, field
import functools
import re
from collections import defaultdict
from typing import Dict, Iterator, List, Match, Optional, Set, Tuple, Union

from pycparser import c_ast as ca
from pycparser.c_ast import ArrayDecl, FuncDecl, IdentifierType, PtrDecl, TypeDecl
from pycparser.c_generator import CGenerator
from pycparser.c_parser import CParser
from pycparser.plyparser import ParseError

from .error import DecompFailure

CType = Union[PtrDecl, ArrayDecl, TypeDecl, FuncDecl]
StructUnion = Union[ca.Struct, ca.Union]
SimpleType = Union[PtrDecl, TypeDecl]


@dataclass
class StructField:
    type: CType
    size: int
    name: str


@dataclass
class Struct:
    type: CType
    fields: Dict[int, List[StructField]]
    # TODO: bitfields
    has_bitfields: bool
    size: int
    align: int


@dataclass
class Array:
    subtype: "DetailedStructMember"
    subctype: CType
    subsize: int
    dim: int


DetailedStructMember = Union[Array, Struct, None]


@dataclass
class Param:
    type: CType
    name: Optional[str]


@dataclass
class Function:
    type: CType
    ret_type: Optional[CType]
    params: Optional[List[Param]]
    is_variadic: bool


@dataclass
class TypeMap:
    typedefs: Dict[str, CType] = field(default_factory=dict)
    var_types: Dict[str, CType] = field(default_factory=dict)
    functions: Dict[str, Function] = field(default_factory=dict)
    structs: Dict[Union[str, int], Struct] = field(default_factory=dict)
    struct_typedefs: Dict[Union[str, int], CType] = field(default_factory=dict)
    enum_values: Dict[str, int] = field(default_factory=dict)


def to_c(node: ca.Node) -> str:
    return CGenerator().visit(node)


def basic_type(names: List[str]) -> TypeDecl:
    idtype = IdentifierType(names=names)
    return TypeDecl(declname=None, quals=[], type=idtype)


def pointer(type: CType) -> CType:
    return PtrDecl(quals=[], type=type)


def resolve_typedefs(type: CType, typemap: TypeMap) -> CType:
    while (
        isinstance(type, TypeDecl)
        and isinstance(type.type, IdentifierType)
        and len(type.type.names) == 1
        and type.type.names[0] in typemap.typedefs
    ):
        type = typemap.typedefs[type.type.names[0]]
    return type


def pointer_decay(type: CType, typemap: TypeMap) -> SimpleType:
    real_type = resolve_typedefs(type, typemap)
    if isinstance(real_type, ArrayDecl):
        return PtrDecl(quals=[], type=real_type.type)
    if isinstance(real_type, FuncDecl):
        return PtrDecl(quals=[], type=type)
    if isinstance(real_type, TypeDecl) and isinstance(real_type.type, ca.Enum):
        return basic_type(["int"])
    assert not isinstance(
        type, (ArrayDecl, FuncDecl)
    ), "resolve_typedefs can't hide arrays/functions"
    return type


def type_from_global_decl(decl: ca.Decl) -> CType:
    """Get the CType of a global Decl, stripping names of function parameters."""
    tp = decl.type
    if not isinstance(tp, ca.FuncDecl) or not tp.args:
        return tp

    def anonymize_param(param: ca.Decl) -> ca.Typename:
        param = copy.deepcopy(param)
        param.name = None
        set_decl_name(param)
        return ca.Typename(name=None, quals=param.quals, type=param.type)

    new_params: List[Union[ca.Decl, ca.ID, ca.Typename, ca.EllipsisParam]] = [
        anonymize_param(param) if isinstance(param, ca.Decl) else param
        for param in tp.args.params
    ]
    return ca.FuncDecl(args=ca.ParamList(new_params), type=tp.type)


def deref_type(type: CType, typemap: TypeMap) -> CType:
    type = resolve_typedefs(type, typemap)
    assert isinstance(type, (ArrayDecl, PtrDecl)), "dereferencing non-pointer"
    return type.type


def is_void(type: CType) -> bool:
    return (
        isinstance(type, ca.TypeDecl)
        and isinstance(type.type, ca.IdentifierType)
        and type.type.names == ["void"]
    )


def equal_types(a: CType, b: CType) -> bool:
    def equal(a: object, b: object) -> bool:
        if a is b:
            return True
        if type(a) != type(b):
            return False
        if a is None:
            return b is None
        if isinstance(a, list):
            assert isinstance(b, list)
            if len(a) != len(b):
                return False
            for i in range(len(a)):
                if not equal(a[i], b[i]):
                    return False
            return True
        if isinstance(a, (int, str)):
            return bool(a == b)
        assert isinstance(a, ca.Node)
        for name in a.__slots__[:-2]:  # type: ignore
            if name == "declname":
                continue
            if not equal(getattr(a, name), getattr(b, name)):
                return False
        return True

    return equal(a, b)


def primitive_size(type: Union[ca.Enum, ca.IdentifierType]) -> int:
    if isinstance(type, ca.Enum):
        return 4
    names = type.names
    if "double" in names:
        return 8
    if "float" in names:
        return 4
    if "short" in names:
        return 2
    if "char" in names:
        return 1
    if "void" in names:
        return 0
    if names.count("long") == 2:
        return 8
    return 4


def function_arg_size_align(type: CType, typemap: TypeMap) -> Tuple[int, int]:
    type = resolve_typedefs(type, typemap)
    if isinstance(type, PtrDecl) or isinstance(type, ArrayDecl):
        return 4, 4
    assert not isinstance(type, FuncDecl), "Function argument can not be a function"
    inner_type = type.type
    if isinstance(inner_type, (ca.Struct, ca.Union)):
        struct = get_struct(inner_type, typemap)
        assert (
            struct is not None
        ), "Function argument can not be of an incomplete struct"
        return struct.size, struct.align
    size = primitive_size(inner_type)
    if size == 0:
        raise DecompFailure("Function parameter has void type")
    return size, size


def var_size_align(type: CType, typemap: TypeMap) -> Tuple[int, int]:
    size, align, _ = parse_struct_member(type, "", typemap, allow_unsized=True)
    return size, align


def is_struct_type(type: CType, typemap: TypeMap) -> bool:
    type = resolve_typedefs(type, typemap)
    if not isinstance(type, TypeDecl):
        return False
    return isinstance(type.type, (ca.Struct, ca.Union))


def get_primitive_list(type: CType, typemap: TypeMap) -> Optional[List[str]]:
    type = resolve_typedefs(type, typemap)
    if not isinstance(type, TypeDecl):
        return None
    inner_type = type.type
    if isinstance(inner_type, ca.Enum):
        return ["int"]
    if isinstance(inner_type, ca.IdentifierType):
        return inner_type.names
    return None


def parse_function(fn: CType) -> Optional[Function]:
    if not isinstance(fn, FuncDecl):
        return None
    params: List[Param] = []
    is_variadic = False
    has_void = False
    if fn.args:
        for arg in fn.args.params:
            if isinstance(arg, ca.EllipsisParam):
                is_variadic = True
            elif isinstance(arg, ca.Decl):
                params.append(Param(type=arg.type, name=arg.name))
            elif isinstance(arg, ca.ID):
                raise DecompFailure(
                    "K&R-style function header is not supported: " + to_c(fn)
                )
            else:
                assert isinstance(arg, ca.Typename)
                if is_void(arg.type):
                    has_void = True
                else:
                    params.append(Param(type=arg.type, name=None))
    maybe_params: Optional[List[Param]] = params
    if not params and not has_void and not is_variadic:
        # Function declaration without a parameter list
        maybe_params = None
    ret_type = None if is_void(fn.type) else fn.type
    return Function(
        type=fn, ret_type=ret_type, params=maybe_params, is_variadic=is_variadic
    )


def divmod_towards_zero(lhs: int, rhs: int, op: str) -> int:
    if rhs < 0:
        rhs = -rhs
        lhs = -lhs
    if lhs < 0:
        return -divmod_towards_zero(-lhs, rhs, op)
    if op == "/":
        return lhs // rhs
    else:
        return lhs % rhs


def parse_constant_int(expr: "ca.Expression", typemap: TypeMap) -> int:
    if isinstance(expr, ca.Constant):
        try:
            return int(expr.value.rstrip("lLuU"), 0)
        except ValueError:
            raise DecompFailure(f"Failed to parse {to_c(expr)} as an int literal")
    if isinstance(expr, ca.ID):
        if expr.name in typemap.enum_values:
            return typemap.enum_values[expr.name]
    if isinstance(expr, ca.BinaryOp):
        op = expr.op
        lhs = parse_constant_int(expr.left, typemap)
        if op == "&&" and lhs == 0:
            return 0
        if op == "||" and lhs != 0:
            return 1
        rhs = parse_constant_int(expr.right, typemap)
        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "<<":
            return lhs << rhs
        if op == ">>":
            return lhs >> rhs
        if op == "&":
            return lhs & rhs
        if op == "|":
            return lhs | rhs
        if op == "^":
            return lhs ^ rhs
        if op == ">=":
            return 1 if lhs >= rhs else 0
        if op == "<=":
            return 1 if lhs <= rhs else 0
        if op == ">":
            return 1 if lhs > rhs else 0
        if op == "<":
            return 1 if lhs < rhs else 0
        if op == "==":
            return 1 if lhs == rhs else 0
        if op == "!=":
            return 1 if lhs != rhs else 0
        if op in ["&&", "||"]:
            return 1 if rhs != 0 else 0
        if op in ["/", "%"]:
            if rhs == 0:
                raise DecompFailure(
                    f"Division by zero when evaluating expression {to_c(expr)}"
                )
            return divmod_towards_zero(lhs, rhs, op)
    if isinstance(expr, ca.TernaryOp):
        cond = parse_constant_int(expr.cond, typemap) != 0
        return parse_constant_int(expr.iftrue if cond else expr.iffalse, typemap)
    if isinstance(expr, ca.ExprList) and not isinstance(expr.exprs[-1], ca.Typename):
        return parse_constant_int(expr.exprs[-1], typemap)
    if isinstance(expr, ca.UnaryOp) and not isinstance(expr.expr, ca.Typename):
        sub = parse_constant_int(expr.expr, typemap)
        if expr.op == "-":
            return -sub
        if expr.op == "~":
            return ~sub
        if expr.op == "!":
            return 1 if sub == 0 else 1
    raise DecompFailure(
        f"Failed to evaluate expression {to_c(expr)} at compile time; only simple arithmetic is supported for now"
    )


def parse_enum(enum: ca.Enum, typemap: TypeMap) -> None:
    """Parse an enum and compute the values of all its enumerators, for use in
    constant evaluation.

    We match IDO in treating all enums as having size 4, so no need to compute
    size or alignment here."""
    if enum.values is None:
        return
    next_value = 0
    for enumerator in enum.values.enumerators:
        if enumerator.value:
            value = parse_constant_int(enumerator.value, typemap)
        else:
            value = next_value
        next_value = value + 1
        typemap.enum_values[enumerator.name] = value


def get_struct(
    struct: Union[ca.Struct, ca.Union], typemap: TypeMap
) -> Optional[Struct]:
    if struct.name:
        return typemap.structs.get(struct.name)
    else:
        return typemap.structs.get(id(struct))


def parse_struct(struct: Union[ca.Struct, ca.Union], typemap: TypeMap) -> Struct:
    existing = get_struct(struct, typemap)
    if existing:
        return existing
    if struct.decls is None:
        raise DecompFailure(f"Tried to use struct {struct.name} before it is defined.")
    ret = do_parse_struct(struct, typemap)
    if struct.name:
        typemap.structs[struct.name] = ret
    typemap.structs[id(struct)] = ret
    return ret


def parse_struct_member(
    type: CType, field_name: str, typemap: TypeMap, *, allow_unsized: bool
) -> Tuple[int, int, DetailedStructMember]:
    old_type = type
    type = resolve_typedefs(type, typemap)
    if isinstance(type, PtrDecl):
        return 4, 4, None
    if isinstance(type, ArrayDecl):
        if type.dim is None:
            raise DecompFailure(f"Array field {field_name} must have a size")
        dim = parse_constant_int(type.dim, typemap)
        size, align, substr = parse_struct_member(
            type.type, field_name, typemap, allow_unsized=False
        )
        return size * dim, align, Array(substr, type.type, size, dim)
    if isinstance(type, FuncDecl):
        assert allow_unsized, "Struct can not contain a function"
        return 0, 0, None
    inner_type = type.type
    if isinstance(inner_type, (ca.Struct, ca.Union)):
        substr = parse_struct(inner_type, typemap)
        return substr.size, substr.align, substr
    if isinstance(inner_type, ca.Enum):
        parse_enum(inner_type, typemap)
    # Otherwise it has to be of type Enum or IdentifierType
    size = primitive_size(inner_type)
    if size == 0 and not allow_unsized:
        raise DecompFailure(f"Field {field_name} cannot be void")
    return size, size, None


def expand_detailed_struct_member(
    substr: DetailedStructMember, type: CType, size: int
) -> Iterator[Tuple[int, str, CType, int]]:
    yield (0, "", type, size)
    if isinstance(substr, Struct):
        for off, sfields in substr.fields.items():
            for field in sfields:
                yield (off, "." + field.name, field.type, field.size)
    elif isinstance(substr, Array) and substr.subsize != 1:
        for i in range(substr.dim):
            for (off, path, subtype, subsize) in expand_detailed_struct_member(
                substr.subtype, substr.subctype, substr.subsize
            ):
                yield (substr.subsize * i + off, f"[{i}]" + path, subtype, subsize)


def do_parse_struct(struct: Union[ca.Struct, ca.Union], typemap: TypeMap) -> Struct:
    is_union = isinstance(struct, ca.Union)
    assert struct.decls is not None, "enforced by caller"
    assert struct.decls, "Empty structs are not valid C"

    fields: Dict[int, List[StructField]] = defaultdict(list)
    union_size = 0
    align = 1
    offset = 0
    bit_offset = 0
    has_bitfields = False
    for decl in struct.decls:
        if not isinstance(decl, ca.Decl):
            continue
        field_name = f"{struct.name}.{decl.name}"
        type = decl.type

        if decl.bitsize is not None:
            # A bitfield "type a : b;" has the following effects on struct layout:
            # - align the struct as if it contained a 'type' field.
            # - allocate the next 'b' bits of the struct, going from high bits to low
            #   within each byte.
            # - ensure that 'a' can be loaded using a single load of the size given by
            #   'type' (lw/lh/lb, unsigned counterparts). If it straddles a 'type'
            #   alignment boundary, skip all bits up to that boundary and then use the
            #   next 'b' bits from there instead.
            has_bitfields = True
            width = parse_constant_int(decl.bitsize, typemap)
            ssize, salign, substr = parse_struct_member(
                type, field_name, typemap, allow_unsized=False
            )
            align = max(align, salign)
            if width == 0:
                continue
            if ssize != salign or substr is not None:
                raise DecompFailure(f"Bitfield {field_name} is not of primitive type")
            if width > ssize * 8:
                raise DecompFailure(f"Width of bitfield {field_name} exceeds its type")
            if is_union:
                union_size = max(union_size, ssize)
            else:
                if offset // ssize != (offset + (bit_offset + width - 1) // 8) // ssize:
                    bit_offset = 0
                    offset = (offset + ssize) & -ssize
                bit_offset += width
                offset += bit_offset // 8
                bit_offset &= 7
            continue

        if not is_union and bit_offset != 0:
            bit_offset = 0
            offset += 1

        if decl.name is not None:
            ssize, salign, substr = parse_struct_member(
                type, field_name, typemap, allow_unsized=False
            )
            align = max(align, salign)
            offset = (offset + salign - 1) & -salign
            for off, path, ftype, fsize in expand_detailed_struct_member(
                substr, type, ssize
            ):
                fields[offset + off].append(
                    StructField(type=ftype, size=fsize, name=decl.name + path)
                )
            if is_union:
                union_size = max(union_size, ssize)
            else:
                offset += ssize
        elif isinstance(type, (ca.Struct, ca.Union)) and type.decls is not None:
            substr = parse_struct(type, typemap)
            if type.name is not None:
                # Struct defined within another, which is silly but valid C.
                # parse_struct already makes sure it gets defined in the global
                # namespace, so no more to do here.
                pass
            else:
                # C extension: anonymous struct/union, whose members are flattened
                align = max(align, substr.align)
                offset = (offset + substr.align - 1) & -substr.align
                for off, sfields in substr.fields.items():
                    for field in sfields:
                        fields[offset + off].append(field)
                if is_union:
                    union_size = max(union_size, substr.size)
                else:
                    offset += substr.size
        elif isinstance(type, ca.Enum):
            parse_enum(type, typemap)

    if not is_union and bit_offset != 0:
        bit_offset = 0
        offset += 1

    # If there is a typedef for this struct, prefer using that name
    if id(struct) in typemap.struct_typedefs:
        ctype = typemap.struct_typedefs[id(struct)]
    elif struct.name and struct.name in typemap.struct_typedefs:
        ctype = typemap.struct_typedefs[struct.name]
    else:
        ctype = TypeDecl(declname=None, quals=[], type=struct)

    size = union_size if is_union else offset
    size = (size + align - 1) & -align
    return Struct(
        type=ctype, fields=fields, has_bitfields=has_bitfields, size=size, align=align
    )


def add_builtin_typedefs(source: str) -> str:
    """Add built-in typedefs to the source code (mips_to_c emits those, so it makes
    sense to pre-define them to simplify hand-written C contexts)."""
    typedefs = {
        "u8": "unsigned char",
        "s8": "char",
        "u16": "unsigned short",
        "s16": "short",
        "u32": "unsigned int",
        "s32": "int",
        "u64": "unsigned long long",
        "s64": "long long",
        "f32": "float",
        "f64": "double",
    }
    line = " ".join(f"typedef {v} {k};" for k, v in typedefs.items())
    return line + "\n" + source


def strip_comments(text: str) -> str:
    # https://stackoverflow.com/a/241506
    def replacer(match: Match[str]) -> str:
        s = match.group(0)
        if s.startswith("/"):
            return " " + "\n" * s.count("\n")
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


def parse_c(source: str) -> ca.FileAST:
    try:
        return CParser().parse(source, "<source>")
    except ParseError as e:
        msg = str(e)
        position, msg = msg.split(": ", 1)
        parts = position.split(":")
        if len(parts) >= 2:
            # Adjust the line number by 1 to correct for the added typedefs
            lineno = int(parts[1]) - 1
            posstr = f" at line {lineno}"
            if len(parts) >= 3:
                posstr += f", column {parts[2]}"
            try:
                line = source.split("\n")[lineno].rstrip()
                posstr += "\n\n" + line
            except IndexError:
                posstr += "(out of bounds?)"
        else:
            posstr = ""
        raise DecompFailure(f"Syntax error when parsing C context.\n{msg}{posstr}")


@functools.lru_cache(maxsize=4)
def build_typemap(source: str) -> TypeMap:
    source = add_builtin_typedefs(source)
    source = strip_comments(source)
    ast: ca.FileAST = parse_c(source)
    ret = TypeMap()

    for item in ast.ext:
        if isinstance(item, ca.Typedef):
            ret.typedefs[item.name] = item.type
            if isinstance(item.type, TypeDecl) and isinstance(
                item.type.type, (ca.Struct, ca.Union)
            ):
                typedef = basic_type([item.name])
                if item.type.type.name:
                    ret.struct_typedefs[item.type.type.name] = typedef
                ret.struct_typedefs[id(item.type.type)] = typedef
        if isinstance(item, ca.FuncDef):
            assert item.decl.name is not None, "cannot define anonymous function"
            fn = parse_function(item.decl.type)
            assert fn is not None
            ret.functions[item.decl.name] = fn
        if isinstance(item, ca.Decl) and isinstance(item.type, FuncDecl):
            assert item.name is not None, "cannot define anonymous function"
            fn = parse_function(item.type)
            assert fn is not None
            ret.functions[item.name] = fn

    defined_function_decls: Set[ca.Decl] = set()

    class Visitor(ca.NodeVisitor):
        def visit_Struct(self, struct: ca.Struct) -> None:
            if struct.decls is not None:
                parse_struct(struct, ret)

        def visit_Union(self, union: ca.Union) -> None:
            if union.decls is not None:
                parse_struct(union, ret)

        def visit_Decl(self, decl: ca.Decl) -> None:
            if decl.name is not None:
                ret.var_types[decl.name] = type_from_global_decl(decl)
            if not isinstance(decl.type, FuncDecl):
                self.visit(decl.type)

        def visit_Enum(self, enum: ca.Enum) -> None:
            parse_enum(enum, ret)

        def visit_FuncDef(self, fn: ca.FuncDef) -> None:
            if fn.decl.name is not None:
                ret.var_types[fn.decl.name] = type_from_global_decl(fn.decl)

    Visitor().visit(ast)
    return ret


def set_decl_name(decl: ca.Decl) -> None:
    name = decl.name
    type = decl.type
    while not isinstance(type, TypeDecl):
        type = type.type
    type.declname = name


def type_to_string(type: CType, name: str = "") -> str:
    if isinstance(type, TypeDecl) and isinstance(
        type.type, (ca.Struct, ca.Union, ca.Enum)
    ):
        if isinstance(type.type, ca.Struct):
            su = "struct"
        else:
            # (ternary to work around a mypy bug)
            su = "union" if isinstance(type.type, ca.Union) else "enum"
        if type.type.name:
            return f"{su} {type.type.name}"
        else:
            return f"anon {su}"
    decl = ca.Decl(name, [], [], [], copy.deepcopy(type), None, None)
    set_decl_name(decl)
    return to_c(decl)


def dump_typemap(typemap: TypeMap) -> None:
    print("Variables:")
    for var, type in typemap.var_types.items():
        print(f"{type_to_string(type, var)};")
    print()
    print("Functions:")
    for name, fn in typemap.functions.items():
        print(f"{type_to_string(fn.type, name)};")
    print()
    print("Structs:")
    for name_or_id, struct in typemap.structs.items():
        if not isinstance(name_or_id, str):
            continue
        print(f"{name_or_id}: size {struct.size}, align {struct.align}")
        for offset, fields in struct.fields.items():
            print(f"  {hex(offset)}:", end="")
            for field in fields:
                print(f" {field.name} ({type_to_string(field.type)})", end="")
            print()
    print()
    print("Enums:")
    for name, value in typemap.enum_values.items():
        print(f"{name}: {value}")
    print()
