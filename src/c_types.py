"""This file handles variable types, function signatures and struct layouts
based on a C AST. Based on the pycparser library."""

from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional, Union
import sys

import attr
from pycparser import c_ast as ca
from pycparser.c_ast import ArrayDecl, TypeDecl, PtrDecl, FuncDecl, IdentifierType
from pycparser.c_generator import CGenerator
from pycparser.c_parser import CParser

from .error import DecompFailure

Type = Union[PtrDecl, ArrayDecl, TypeDecl, FuncDecl]
SimpleType = Union[PtrDecl, TypeDecl]


@attr.s
class StructField:
    type: Type = attr.ib()
    name: str = attr.ib()


@attr.s
class Struct:
    fields: Dict[int, List[StructField]] = attr.ib()
    # TODO: bitfields
    size: int = attr.ib()
    align: int = attr.ib()


@attr.s
class Param:
    type: Type = attr.ib()
    name: Optional[str] = attr.ib()


@attr.s
class Function:
    ret_type: Optional[Type] = attr.ib()
    params: List[Param] = attr.ib()
    is_variadic: bool = attr.ib()


@attr.s
class TypeMap:
    typedefs: Dict[str, Type] = attr.ib(factory=dict)
    var_types: Dict[str, Type] = attr.ib(factory=dict)
    functions: Dict[str, Function] = attr.ib(factory=dict)
    struct_defs: Dict[str, Struct] = attr.ib(factory=dict)


def to_c(node: ca.Node) -> str:
    return CGenerator().visit(node)


def basic_type(name: str) -> TypeDecl:
    idtype = IdentifierType(names=[name])
    return TypeDecl(declname=None, quals=[], type=idtype)


def pointer(type: Type) -> Type:
    return PtrDecl(quals=[], type=type)


def resolve_typedefs(type: Type, typemap: TypeMap) -> Type:
    while (
        isinstance(type, TypeDecl)
        and isinstance(type.type, IdentifierType)
        and len(type.type.names) == 1
        and type.type.names[0] in typemap.typedefs
    ):
        type = typemap.typedefs[type.type.names[0]]
    return type


def pointer_decay(type: Type, typemap: TypeMap) -> SimpleType:
    real_type = resolve_typedefs(type, typemap)
    if isinstance(real_type, ArrayDecl):
        return PtrDecl(quals=[], type=real_type.type)
    if isinstance(real_type, FuncDecl):
        return PtrDecl(quals=[], type=type)
    if isinstance(real_type, TypeDecl) and isinstance(real_type.type, ca.Enum):
        return basic_type("int")
    assert not isinstance(
        type, (ArrayDecl, FuncDecl)
    ), "resolve_typedefs can't hide arrays/functions"
    return type


def deref_type(type: Type, typemap: TypeMap) -> Type:
    type = resolve_typedefs(type, typemap)
    assert isinstance(type, (ArrayDecl, PtrDecl)), "dereferencing non-pointer"
    return type.type


def is_void(type: Type) -> bool:
    return (
        isinstance(type, ca.TypeDecl)
        and isinstance(type.type, ca.IdentifierType)
        and type.type.names == ["void"]
    )


def parse_function(fn: FuncDecl) -> Function:
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
    if not params and not has_void:
        is_variadic = True
    ret_type = None if is_void(fn.type) else fn.type
    return Function(ret_type=ret_type, params=params, is_variadic=is_variadic)


def parse_constant_int(expr: "ca.Expression") -> int:
    if isinstance(expr, ca.Constant):
        try:
            return int(expr.value.rstrip("lLuU"), 0)
        except ValueError:
            raise DecompFailure(f"Failed to parse {to_c(expr)} as an int literal")
    if isinstance(expr, ca.BinaryOp):
        lhs = parse_constant_int(expr.left)
        rhs = parse_constant_int(expr.right)
        if expr.op == "+":
            return lhs + rhs
        if expr.op == "-":
            return lhs - rhs
        if expr.op == "*":
            return lhs * rhs
        if expr.op == "<<":
            return lhs << rhs
        if expr.op == ">>":
            return lhs >> rhs
    raise DecompFailure(
        f"Failed to evaluate expression {to_c(expr)} at compile time; only simple arithmetic is supported for now"
    )


def parse_struct(struct: Union[ca.Struct, ca.Union], typemap: TypeMap) -> Struct:
    is_union = isinstance(struct, ca.Union)
    assert (
        struct.decls is not None
    ), "parse_struct is only called on structs with .decls"
    assert struct.decls, "Empty structs are not valid C"

    def parse_struct_member(
        type: Type, field_name: str
    ) -> Tuple[int, int, Optional[Struct]]:
        type = resolve_typedefs(type, typemap)
        if isinstance(type, PtrDecl):
            return 4, 4, None
        if isinstance(type, ArrayDecl):
            if type.dim is None:
                raise DecompFailure(f"Array field {field_name} must have a size")
            dim = parse_constant_int(type.dim)
            size, align, _ = parse_struct_member(type.type, field_name)
            return size * dim, align, None
        assert not isinstance(type, FuncDecl), "Struct can not contain a function"
        inner_type = type.type
        if isinstance(inner_type, (ca.Struct, ca.Union)):
            if inner_type.name is not None and inner_type.name in typemap.struct_defs:
                substr = typemap.struct_defs[inner_type.name]
                return substr.size, substr.align, substr
            if inner_type.decls is not None:
                substr = parse_struct(inner_type, typemap)
                if inner_type.name is not None:
                    typemap.struct_defs[inner_type.name] = substr
                return substr.size, substr.align, substr
            raise DecompFailure(
                f"Field {field_name} is of undefined struct type {inner_type.name}"
            )
        if isinstance(inner_type, ca.Enum):
            return 4, 4, None
        # Otherwise it has to be of type IdentifierType
        if "double" in inner_type.names:
            return 8, 8, None
        if "float" in inner_type.names:
            return 4, 4, None
        if "short" in inner_type.names:
            return 2, 2, None
        if "char" in inner_type.names:
            return 1, 1, None
        if inner_type.names.count("long") == 2:
            return 8, 8, None
        return 4, 4, None

    fields: Dict[int, List[StructField]] = defaultdict(list)
    union_size = 0
    align = 1
    offset = 0
    bit_offset = 0
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
            width = parse_constant_int(decl.bitsize)
            ssize, salign, substr = parse_struct_member(type, field_name)
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
            ssize, salign, substr = parse_struct_member(type, field_name)
            align = max(align, salign)
            offset = (offset + salign - 1) & -salign
            fields[offset].append(StructField(type=type, name=decl.name))
            if substr is not None:
                for off, sfields in substr.fields.items():
                    for field in sfields:
                        fields[offset + off].append(
                            StructField(
                                type=field.type, name=decl.name + "." + field.name
                            )
                        )
            if is_union:
                union_size = max(union_size, ssize)
            else:
                offset += ssize
        elif isinstance(type, (ca.Struct, ca.Union)) and type.decls is not None:
            if type.name is None:
                # C extension: anonymous struct/union, whose members are flattened
                substr = parse_struct(type, typemap)
                align = max(align, substr.align)
                offset = (offset + substr.align - 1) & -substr.align
                for off, sfields in substr.fields.items():
                    for field in sfields:
                        fields[offset + off].append(field)
                if is_union:
                    union_size = max(union_size, substr.size)
                else:
                    offset += substr.size
            else:
                # Struct defined within another. Silly but valid C.
                substr = parse_struct(type, typemap)
                typemap.struct_defs[type.name] = substr

    if not is_union and bit_offset != 0:
        bit_offset = 0
        offset += 1

    size = union_size if is_union else (offset + align - 1) & -align
    return Struct(fields=fields, size=size, align=align)


def build_typemap(source: str, filename: str = "") -> TypeMap:
    ast: ca.FileAST = CParser().parse(source, filename)
    ret = TypeMap()
    for item in ast.ext:
        if isinstance(item, ca.Typedef):
            ret.typedefs[item.name] = item.type
        if isinstance(item, ca.FuncDef):
            assert item.decl.name is not None, "cannot define anonymous function"
            assert isinstance(item.decl.type, FuncDecl)
            ret.functions[item.decl.name] = parse_function(item.decl.type)
        if isinstance(item, ca.Decl) and isinstance(item.type, FuncDecl):
            assert item.name is not None, "cannot define anonymous function"
            ret.functions[item.name] = parse_function(item.type)

    defined_function_decls: Set[ca.Decl] = set()

    class Visitor(ca.NodeVisitor):
        def visit_Struct(self, struct: ca.Struct) -> None:
            if struct.decls and struct.name is not None and struct.decls is not None:
                ret.struct_defs[struct.name] = parse_struct(struct, ret)

        def visit_Union(self, union: ca.Union) -> None:
            if union.decls and union.name is not None and union.decls is not None:
                ret.struct_defs[union.name] = parse_struct(union, ret)

        def visit_Decl(self, decl: ca.Decl) -> None:
            if decl.name is not None:
                ret.var_types[decl.name] = decl.type
            if not isinstance(decl.type, FuncDecl):
                self.visit(decl.type)

        def visit_Enum(self, enum: ca.Enum) -> None:
            if enum.name is not None:
                ret.typedefs[enum.name] = basic_type("int")

        def visit_FuncDef(self, fn: ca.FuncDef) -> None:
            if fn.decl.name is not None:
                ret.var_types[fn.decl.name] = fn.decl.type

    Visitor().visit(ast)
    return ret


def set_decl_name(decl: ca.Decl) -> None:
    name = decl.name
    type = decl.type
    while not isinstance(type, TypeDecl):
        type = type.type
    type.declname = name


def type_to_string(type: Type) -> str:
    if isinstance(type, TypeDecl) and isinstance(type.type, (ca.Struct, ca.Union)):
        su = "struct" if isinstance(type.type, ca.Struct) else "union"
        return type.type.name or f"anon {su}"
    else:
        decl = ca.Decl("", [], [], [], type, None, None)
        set_decl_name(decl)
        return to_c(decl)


def dump_typemap(typemap: TypeMap) -> None:
    print("Variables:")
    for var, type in typemap.var_types.items():
        print(f"{var}:", type_to_string(type))
    print()
    print("Functions:")
    for name, fn in typemap.functions.items():
        params = [type_to_string(arg.type) for arg in fn.params]
        if fn.is_variadic:
            params.append("...")
        ret_str = "void" if fn.ret_type is None else type_to_string(fn.ret_type)
        print(f"{name}: {ret_str}({', '.join(params)})")
    print()
    print("Structs:")
    for name, struct in typemap.struct_defs.items():
        print(f"{name}: size {struct.size}, align {struct.align}")
        for offset, fields in struct.fields.items():
            print(f"  {offset}:", end="")
            for field in fields:
                print(f" {field.name} ({type_to_string(field.type)})", end="")
            print()
    print()
