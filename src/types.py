import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import pycparser.c_ast as ca

from .c_types import (
    CType,
    Struct,
    StructUnion as CStructUnion,
    TypeMap,
    parse_constant_int,
    parse_function,
    parse_struct,
    primitive_size,
    resolve_typedefs,
    set_decl_name,
    to_c,
)
from .error import DecompFailure, static_assert_unreachable
from .options import Formatter

# AccessPath represents a struct/array path, with ints for array access, and
# strs for struct fields. Ex: `["foo", 3, "bar"]` represents `.foo[3].bar`
AccessPath = List[Union[str, int]]


@dataclass
class TypePool:
    """
    Mutable shared state for Types, maintaining the set of available struct types.
    """

    structs: Set["StructDeclaration"] = field(default_factory=set)
    structs_by_tag_name: Dict[str, "StructDeclaration"] = field(default_factory=dict)
    structs_by_ctype: Dict[CStructUnion, "StructDeclaration"] = field(
        default_factory=dict
    )

    def get_struct_for_ctype(
        self,
        ctype: CStructUnion,
    ) -> Optional["StructDeclaration"]:
        """Return the StructDeclaration for a given ctype struct, if known"""
        struct = self.structs_by_ctype.get(ctype)
        if struct is not None:
            return struct
        if ctype.name is not None:
            return self.structs_by_tag_name.get(ctype.name)
        return None

    def add_struct(
        self,
        struct: "StructDeclaration",
        ctype_or_tag_name: Union[CStructUnion, str],
    ) -> None:
        """Add the struct declaration to the set of known structs for later access"""
        self.structs.add(struct)

        tag_name: Optional[str]
        if isinstance(ctype_or_tag_name, str):
            tag_name = ctype_or_tag_name
        else:
            ctype = ctype_or_tag_name
            tag_name = ctype.name
            self.structs_by_ctype[ctype] = struct

        if tag_name is not None:
            assert (
                tag_name not in self.structs_by_tag_name
            ), f"Duplicate tag: {tag_name}"
            self.structs_by_tag_name[tag_name] = struct


@dataclass(eq=False)
class TypeData:
    K_INT = 1 << 0
    K_PTR = 1 << 1
    K_FLOAT = 1 << 2
    K_FN = 1 << 3
    K_VOID = 1 << 4
    K_ARRAY = 1 << 5
    K_STRUCT = 1 << 6
    K_INTPTR = K_INT | K_PTR
    K_ANYREG = K_INT | K_PTR | K_FLOAT
    K_ANY = K_INT | K_PTR | K_FLOAT | K_FN | K_VOID | K_ARRAY | K_STRUCT

    SIGNED = 1
    UNSIGNED = 2
    ANY_SIGN = 3

    kind: int = K_ANY
    likely_kind: int = K_ANY  # subset of kind
    size_bits: Optional[int] = None
    uf_parent: Optional["TypeData"] = None

    sign: int = ANY_SIGN  # K_INT
    ptr_to: Optional["Type"] = None  # K_PTR | K_ARRAY
    fn_sig: Optional["FunctionSignature"] = None  # K_FN
    array_dim: Optional[int] = None  # K_ARRAY
    struct: Optional["StructDeclaration"] = None  # K_STRUCT

    def __post_init__(self) -> None:
        assert self.kind
        self.likely_kind &= self.kind

    def get_representative(self) -> "TypeData":
        # Follow `uf_parent` links until we hit the "root" TypeData
        equivalent_typedatas = set()
        root = self
        while root.uf_parent is not None:
            assert root not in equivalent_typedatas, "TypeData cycle detected"
            equivalent_typedatas.add(root)
            root = root.uf_parent

        # Set the `uf_parent` pointer on all visited TypeDatas
        for td in equivalent_typedatas:
            td.uf_parent = root

        return root


@dataclass(eq=False, repr=False)
class Type:
    """
    Type information for an expression, which may improve over time. The least
    specific type is any (initially the case for e.g. arguments); this might
    get refined into intish if the value gets used for e.g. an integer add
    operation, or into u32 if it participates in a logical right shift.
    Types cannot change except for improvements of this kind -- thus concrete
    types like u32 can never change into anything else, and e.g. ints can't
    become floats.
    """

    _data: TypeData

    def unify(self, other: "Type", *, seen: Optional[Set["TypeData"]] = None) -> bool:
        """
        Try to set this type equal to another. Returns true on success.
        Once set equal, the types will always be equal (we use a union-find
        structure to ensure this).
        The seen argument is used during recursion, to track which TypeData
        objects have been encountered so far.
        """

        x = self.data()
        y = other.data()
        if x is y:
            return True

        # If we hit a type that we have already seen, fail.
        # TODO: Is there a looser check that would allow more types to unify?
        if seen is None:
            seen = {x, y}
        elif x in seen or y in seen:
            return False
        else:
            seen = seen | {x, y}

        if (
            x.size_bits is not None
            and y.size_bits is not None
            and x.size_bits != y.size_bits
        ):
            return False

        kind = x.kind & y.kind
        likely_kind = x.likely_kind & y.likely_kind
        size_bits = x.size_bits if x.size_bits is not None else y.size_bits
        ptr_to = x.ptr_to if x.ptr_to is not None else y.ptr_to
        fn_sig = x.fn_sig if x.fn_sig is not None else y.fn_sig
        array_dim = x.array_dim if x.array_dim is not None else y.array_dim
        struct = x.struct if x.struct is not None else y.struct
        sign = x.sign & y.sign
        if size_bits not in (None, 32, 64):
            kind &= ~TypeData.K_FLOAT
        if size_bits not in (None, 32):
            kind &= ~TypeData.K_PTR
        if size_bits not in (None,):
            kind &= ~TypeData.K_FN
        if size_bits not in (None, 0):
            kind &= ~TypeData.K_VOID
        likely_kind &= kind
        if kind == 0 or sign == 0:
            return False
        if kind == TypeData.K_PTR:
            size_bits = 32
        if sign != TypeData.ANY_SIGN:
            assert kind == TypeData.K_INT
        if x.ptr_to is not None and y.ptr_to is not None:
            if not x.ptr_to.unify(y.ptr_to, seen=seen):
                return False
        if x.fn_sig is not None and y.fn_sig is not None:
            if not x.fn_sig.unify(y.fn_sig, seen=seen):
                return False
        if x.array_dim is not None and y.array_dim is not None:
            if x.array_dim != y.array_dim:
                return False
        if x.struct is not None and y.struct is not None:
            if not x.struct.unify(y.struct, seen=seen):
                return False
        x.kind = kind
        x.likely_kind = likely_kind
        x.size_bits = size_bits
        x.sign = sign
        x.ptr_to = ptr_to
        x.fn_sig = fn_sig
        x.array_dim = array_dim
        x.struct = struct
        y.uf_parent = x
        return True

    def data(self) -> "TypeData":
        if self._data.uf_parent is None:
            return self._data
        self._data = self._data.get_representative()
        return self._data

    def is_float(self) -> bool:
        return self.data().kind == TypeData.K_FLOAT

    def is_likely_float(self) -> bool:
        data = self.data()
        return data.kind == TypeData.K_FLOAT or data.likely_kind == TypeData.K_FLOAT

    def is_pointer(self) -> bool:
        return self.data().kind == TypeData.K_PTR

    def is_pointer_or_array(self) -> bool:
        return self.data().kind in (TypeData.K_PTR, TypeData.K_ARRAY)

    def is_int(self) -> bool:
        return self.data().kind == TypeData.K_INT

    def is_reg(self) -> bool:
        return (self.data().kind & ~TypeData.K_ANYREG) == 0

    def is_function(self) -> bool:
        return self.data().kind == TypeData.K_FN

    def is_void(self) -> bool:
        return self.data().kind == TypeData.K_VOID

    def is_array(self) -> bool:
        return self.data().kind == TypeData.K_ARRAY

    def is_struct(self) -> bool:
        return self.data().kind == TypeData.K_STRUCT

    def is_unsigned(self) -> bool:
        return self.data().sign == TypeData.UNSIGNED

    def get_size_bits(self) -> Optional[int]:
        return self.data().size_bits

    def get_size_bytes(self) -> Optional[int]:
        size_bits = self.get_size_bits()
        return None if size_bits is None else size_bits // 8

    def get_parameter_size_align_bytes(self) -> Tuple[int, int]:
        """Return the size & alignment of self when used as a function argument"""
        data = self.data()
        if self.is_struct():
            assert data.struct is not None
            return data.struct.size, data.struct.align
        size = (self.get_size_bits() or 32) // 8
        return size, size

    def get_pointer_target(self) -> Optional["Type"]:
        """If self is a pointer-to-a-Type, return the Type"""
        data = self.data()
        if self.is_pointer() and data.ptr_to is not None:
            return data.ptr_to
        return None

    def reference(self) -> "Type":
        """Return a pointer to self. If self is an array, decay the type to a pointer"""
        if self.is_array():
            data = self.data()
            assert data.ptr_to is not None
            return Type.ptr(data.ptr_to)
        return Type.ptr(self)

    def decay(self) -> "Type":
        """If self is an array, return a pointer to the element type. Otherwise, return self."""
        if self.is_array():
            data = self.data()
            assert data.ptr_to is not None
            return Type.ptr(data.ptr_to)
        return self

    def get_array(self) -> Tuple[Optional["Type"], Optional[int]]:
        """If self is an array, return a tuple of the inner Type & the array dimension"""
        if not self.is_array():
            return None, None
        data = self.data()
        assert data.ptr_to is not None
        return (data.ptr_to, data.array_dim)

    def get_function_pointer_signature(self) -> Optional["FunctionSignature"]:
        """If self is a function pointer, return the FunctionSignature"""
        data = self.data()
        if self.is_pointer() and data.ptr_to is not None:
            ptr_to = data.ptr_to.data()
            if ptr_to.kind == TypeData.K_FN:
                return ptr_to.fn_sig
        return None

    def get_struct_declaration(self) -> Optional["StructDeclaration"]:
        """If self is a struct, return the StructDeclaration"""
        if self.is_struct():
            data = self.data()
            assert data.struct is not None
            return data.struct
        return None

    GetFieldResult = Tuple[Optional[AccessPath], "Type", int]

    def get_field(self, offset: int, *, target_size: Optional[int]) -> GetFieldResult:
        """
        Locate the field in self at the appropriate offset, and optionally
        with an exact target size (both values in bytes).
        The target size can be used to disambiguate different fields in a union, or
        different levels inside nested structs.

        The return value is a tuple of `(field_path, field_type, remaining_offset)`.
        If no field is found, the result is `(None, Type.any(), offset)`.
        If `remaining_offset` is nonzero, then there was *not* a field at the exact
        offset provided; the returned field is at `(offset - remaining_offset)`.
        """
        NO_MATCHING_FIELD: Type.GetFieldResult = (None, Type.any(), offset)

        if offset < 0:
            return NO_MATCHING_FIELD

        if self.is_array():
            # Array types always have elements with known size
            data = self.data()
            assert data.ptr_to is not None
            size = data.ptr_to.get_size_bytes()
            assert size is not None

            index, remaining_offset = divmod(offset, size)
            if data.array_dim is not None and index >= data.array_dim:
                return NO_MATCHING_FIELD
            assert index >= 0 and remaining_offset >= 0

            # Assume this is an array access at the computed `index`.
            # Check if there is a field at the `remaining_offset` offset
            subpath, subtype, sub_remaining_offset = data.ptr_to.get_field(
                remaining_offset, target_size=target_size
            )
            if subpath is not None:
                # Success: prepend `index` and return
                subpath.insert(0, index)
                return subpath, subtype, sub_remaining_offset
            return NO_MATCHING_FIELD

        if self.is_struct():
            data = self.data()
            assert data.struct is not None
            possible_fields = data.struct.fields_at_offset(offset)
            if not possible_fields:
                return NO_MATCHING_FIELD
            possible_results: List[Type.GetFieldResult] = []
            if target_size is None or target_size == self.get_size_bytes():
                possible_results.append(([], self, offset))
            for field in possible_fields:
                inner_offset_bits = offset - field.offset
                subpath, subtype, sub_remaining_offset = field.type.get_field(
                    inner_offset_bits, target_size=target_size
                )
                if subpath is None:
                    continue
                subpath.insert(0, field.name)
                possible_results.append((subpath, subtype, sub_remaining_offset))
                if (
                    target_size is not None
                    and target_size == subtype.get_size_bytes()
                    and sub_remaining_offset == 0
                ):
                    return possible_results[-1]
            zero_offset_results = [r for r in possible_results if r[2] == 0]
            if zero_offset_results:
                return zero_offset_results[0]
            if possible_results:
                return possible_results[0]

        if offset == 0 and (
            target_size is None or target_size == self.get_size_bytes()
        ):
            # The whole type itself is a match
            return [], self, 0

        return NO_MATCHING_FIELD

    def get_deref_field(
        self, offset: int, *, target_size: Optional[int]
    ) -> GetFieldResult:
        """
        Similar to `.get_field()`, but treat self as a pointer and find the field in the
        pointer's target.  The return value has the same semantics as `.get_field()`.

        If successful, the first item in the resulting `field_path` will be `0`.
        This mirrors how `foo[0].bar` and `foo->bar` are equivalent in C.
        """
        NO_MATCHING_FIELD: Type.GetFieldResult = (None, Type.any(), offset)

        target = self.get_pointer_target()
        if target is None:
            return NO_MATCHING_FIELD

        # Assume the pointer is to a single object, and not an array.
        field_path, field_type, remaining_offset = target.get_field(
            offset, target_size=target_size
        )
        if field_path is not None:
            field_path.insert(0, 0)
        return field_path, field_type, remaining_offset

    def get_initializer_fields(
        self,
    ) -> Optional[List[Union[int, "Type"]]]:
        """
        If self is a struct or array (i.e. initialized with `{...}` syntax), then
        return a list of fields suitable for creating an initializer.
        Return None if an initializer cannot be made (e.g. a struct with bitfields)

        Padding is represented by an int in the list, otherwise the list fields
        denote the field's Type.
        """
        data = self.data()
        if self.is_array():
            assert data.ptr_to is not None
            if data.array_dim is None:
                return None

            return [data.ptr_to] * data.array_dim

        if self.is_struct():
            assert data.struct is not None
            if data.struct.has_bitfields:
                # TODO: Support bitfields
                return None

            output: List[Union[int, Type]] = []
            position = 0

            def add_padding(upto: int) -> None:
                nonlocal position
                nonlocal output
                assert upto >= position
                if upto > position:
                    output.append(upto - position)

            for field in data.struct.fields:
                assert field.offset >= position, "overlapping fields"

                add_padding(field.offset)
                field_size = field.type.get_size_bytes()
                assert field_size is not None
                output.append(field.type)
                position = field.offset + field_size

                # Unions only have an initializer for the first field
                if data.struct.is_union:
                    break

            add_padding(data.struct.size)
            return output

        return None

    def to_decl(self, name: str, fmt: Formatter) -> str:
        decl = ca.Decl(
            name=name,
            type=self._to_ctype(set(), fmt),
            quals=[],
            storage=[],
            funcspec=[],
            init=None,
            bitsize=None,
        )
        set_decl_name(decl)
        ret = to_c(decl)

        if fmt.coding_style.pointer_style_left:
            # Keep going until the result is unmodified
            while True:
                replaced = (
                    ret.replace(" *", "* ").replace("* )", "*)").replace("* ,", "*,")
                )
                if replaced == ret:
                    break
                ret = replaced

        return ret

    def _to_ctype(self, seen: Set["TypeData"], fmt: Formatter) -> CType:
        def simple_ctype(typename: str) -> ca.TypeDecl:
            return ca.TypeDecl(
                type=ca.IdentifierType(names=[typename]), declname=None, quals=[]
            )

        unk_symbol = "MIPS2C_UNK" if fmt.valid_syntax else "?"

        data = self.data()
        if data in seen:
            return simple_ctype(unk_symbol)
        seen.add(data)
        size_bits = data.size_bits or 32
        sign = "s" if data.sign & TypeData.SIGNED else "u"

        if (data.kind & TypeData.K_ANYREG) == TypeData.K_ANYREG and (
            data.likely_kind & (TypeData.K_INT | TypeData.K_FLOAT)
        ) not in (TypeData.K_INT, TypeData.K_FLOAT):
            if data.size_bits is not None:
                return simple_ctype(f"{unk_symbol}{size_bits}")
            return simple_ctype(unk_symbol)

        if (
            data.kind == TypeData.K_FLOAT
            or (data.likely_kind & (TypeData.K_FLOAT | TypeData.K_INT))
            == TypeData.K_FLOAT
        ):
            return simple_ctype(f"f{size_bits}")

        if data.kind == TypeData.K_PTR:
            target_ctype: CType
            if data.ptr_to is None:
                target_ctype = simple_ctype("void")
            else:
                target_ctype = data.ptr_to._to_ctype(seen.copy(), fmt)

            # Strip parameter names from function pointers
            if isinstance(target_ctype, ca.FuncDecl) and target_ctype.args:
                for arg in target_ctype.args.params:
                    if isinstance(arg, ca.Decl):
                        arg.name = None
                        set_decl_name(arg)

            return ca.PtrDecl(type=target_ctype, quals=[])

        if data.kind == TypeData.K_FN:
            assert data.fn_sig is not None
            return_ctype = data.fn_sig.return_type._to_ctype(seen.copy(), fmt)

            params: List[Union[ca.Decl, ca.ID, ca.Typename, ca.EllipsisParam]] = []
            for param in data.fn_sig.params:
                decl = ca.Decl(
                    name=param.name,
                    type=param.type._to_ctype(seen.copy(), fmt),
                    quals=[],
                    storage=[],
                    funcspec=[],
                    init=None,
                    bitsize=None,
                )
                set_decl_name(decl)
                params.append(decl)

            if data.fn_sig.is_variadic:
                params.append(ca.EllipsisParam())

            return ca.FuncDecl(
                type=return_ctype,
                args=ca.ParamList(params),
            )

        if data.kind == TypeData.K_VOID:
            return simple_ctype("void")

        if data.kind == TypeData.K_ARRAY:
            assert data.ptr_to is not None
            dim: Optional[ca.Constant] = None
            if data.array_dim is not None:
                dim = ca.Constant(value=str(data.array_dim), type="")
            return ca.ArrayDecl(
                type=data.ptr_to._to_ctype(seen.copy(), fmt),
                dim=dim,
                dim_quals=[],
            )

        if data.kind == TypeData.K_STRUCT:
            assert data.struct is not None
            if data.struct.typedef_name:
                return simple_ctype(data.struct.typedef_name)
            # If there's no typedef or tag name, then label it as `_anonymous`
            name = data.struct.tag_name or "_anonymous"
            Class = ca.Union if data.struct.is_union else ca.Struct
            return ca.TypeDecl(
                declname=name, type=ca.Struct(name=name, decls=None), quals=[]
            )

        return simple_ctype(f"{sign}{size_bits}")

    def format(self, fmt: Formatter) -> str:
        return self.to_decl("", fmt)

    def __str__(self) -> str:
        return self.format(Formatter(debug=True))

    def __repr__(self) -> str:
        data = self.data()
        signstr = ("+" if data.sign & TypeData.SIGNED else "") + (
            "-" if data.sign & TypeData.UNSIGNED else ""
        )
        kindstr = (
            ("I" if data.kind & TypeData.K_INT else "")
            + ("P" if data.kind & TypeData.K_PTR else "")
            + ("F" if data.kind & TypeData.K_FLOAT else "")
            + ("N" if data.kind & TypeData.K_FN else "")
            + ("V" if data.kind & TypeData.K_VOID else "")
            + ("A" if data.kind & TypeData.K_ARRAY else "")
            + ("S" if data.kind & TypeData.K_STRUCT else "")
        )
        sizestr = str(data.size_bits) if data.size_bits is not None else "?"
        return f"Type({signstr + kindstr + sizestr})"

    @staticmethod
    def any() -> "Type":
        return Type(TypeData())

    @staticmethod
    def any_reg() -> "Type":
        return Type(TypeData(kind=TypeData.K_ANYREG))

    @staticmethod
    def intish() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT))

    @staticmethod
    def intptr() -> "Type":
        return Type(TypeData(kind=TypeData.K_INTPTR))

    @staticmethod
    def ptr(type: Optional["Type"] = None) -> "Type":
        return Type(TypeData(kind=TypeData.K_PTR, size_bits=32, ptr_to=type))

    @staticmethod
    def function(fn_sig: Optional["FunctionSignature"] = None) -> "Type":
        if fn_sig is None:
            fn_sig = FunctionSignature()
        return Type(TypeData(kind=TypeData.K_FN, fn_sig=fn_sig))

    @staticmethod
    def f32() -> "Type":
        return Type(TypeData(kind=TypeData.K_FLOAT, size_bits=32))

    @staticmethod
    def floatish() -> "Type":
        return Type(TypeData(kind=TypeData.K_FLOAT))

    @staticmethod
    def f64() -> "Type":
        return Type(TypeData(kind=TypeData.K_FLOAT, size_bits=64))

    @staticmethod
    def s8() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=8, sign=TypeData.SIGNED))

    @staticmethod
    def u8() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=8, sign=TypeData.UNSIGNED))

    @staticmethod
    def s16() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=16, sign=TypeData.SIGNED))

    @staticmethod
    def u16() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=16, sign=TypeData.UNSIGNED))

    @staticmethod
    def s32() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=32, sign=TypeData.SIGNED))

    @staticmethod
    def u32() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=32, sign=TypeData.UNSIGNED))

    @staticmethod
    def s64() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=64, sign=TypeData.SIGNED))

    @staticmethod
    def u64() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=64, sign=TypeData.UNSIGNED))

    @staticmethod
    def int64() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=64))

    @staticmethod
    def int_of_size(size_bits: int) -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size_bits=size_bits))

    @staticmethod
    def reg32(*, likely_float: bool) -> "Type":
        likely = TypeData.K_FLOAT if likely_float else TypeData.K_INTPTR
        return Type(TypeData(kind=TypeData.K_ANYREG, likely_kind=likely, size_bits=32))

    @staticmethod
    def reg64(*, likely_float: bool) -> "Type":
        kind = TypeData.K_FLOAT | TypeData.K_INT
        likely = TypeData.K_FLOAT if likely_float else TypeData.K_INT
        return Type(TypeData(kind=kind, likely_kind=likely, size_bits=64))

    @staticmethod
    def bool() -> "Type":
        return Type.intish()

    @staticmethod
    def void() -> "Type":
        return Type(TypeData(kind=TypeData.K_VOID, size_bits=0))

    @staticmethod
    def array(type: "Type", dim: Optional[int]) -> "Type":
        # Array elements must have a known size
        el_size = type.get_size_bits()
        assert el_size is not None

        size_bits = None if dim is None else (el_size * dim)
        return Type(
            TypeData(
                kind=TypeData.K_ARRAY, size_bits=size_bits, ptr_to=type, array_dim=dim
            )
        )

    @staticmethod
    def struct(st: "StructDeclaration") -> "Type":
        return Type(TypeData(kind=TypeData.K_STRUCT, size_bits=st.size * 8, struct=st))

    @staticmethod
    def ctype(ctype: CType, typemap: TypeMap, typepool: TypePool) -> "Type":
        real_ctype = resolve_typedefs(ctype, typemap)
        if isinstance(real_ctype, ca.ArrayDecl):
            dim = None
            try:
                if real_ctype.dim is not None:
                    dim = parse_constant_int(real_ctype.dim, typemap)
            except DecompFailure:
                pass
            inner_type = Type.ctype(real_ctype.type, typemap, typepool)
            return Type.array(inner_type, dim)
        if isinstance(real_ctype, ca.PtrDecl):
            return Type.ptr(Type.ctype(real_ctype.type, typemap, typepool))
        if isinstance(real_ctype, ca.FuncDecl):
            fn = parse_function(real_ctype)
            assert fn is not None
            fn_sig = FunctionSignature(
                return_type=Type.void(),
                is_variadic=fn.is_variadic,
            )
            if fn.ret_type is not None:
                fn_sig.return_type = Type.ctype(fn.ret_type, typemap, typepool)
            if fn.params is not None:
                fn_sig.params = [
                    FunctionParam(
                        name=param.name or "",
                        type=Type.ctype(param.type, typemap, typepool),
                    )
                    for param in fn.params
                ]
                fn_sig.params_known = True
            return Type.function(fn_sig)
        if isinstance(real_ctype, ca.TypeDecl):
            if isinstance(real_ctype.type, (ca.Struct, ca.Union)):
                return Type.struct(
                    StructDeclaration.from_ctype(real_ctype.type, typemap, typepool)
                )
            names = (
                ["int"]
                if isinstance(real_ctype.type, ca.Enum)
                else real_ctype.type.names
            )
            if "double" in names:
                return Type.f64()
            if "float" in names:
                return Type.f32()
            if "void" in names:
                return Type.void()
            size_bits = 8 * primitive_size(real_ctype.type)
            assert size_bits > 0
            sign = TypeData.UNSIGNED if "unsigned" in names else TypeData.SIGNED
            return Type(TypeData(kind=TypeData.K_INT, size_bits=size_bits, sign=sign))
        static_assert_unreachable(real_ctype)


@dataclass(eq=False)
class FunctionParam:
    type: Type = field(default_factory=Type.any)
    name: str = ""


@dataclass(eq=False)
class FunctionSignature:
    return_type: Type = field(default_factory=Type.any)
    params: List[FunctionParam] = field(default_factory=list)
    params_known: bool = False
    is_variadic: bool = False

    def unify(self, other: "FunctionSignature", *, seen: Set[TypeData]) -> bool:
        if self.params_known and other.params_known:
            if self.is_variadic != other.is_variadic:
                return False
            if len(self.params) != len(other.params):
                return False

        # Try to unify *all* ret/param types, without returning early
        # TODO: If not all the types unify, roll back any changes made
        can_unify = self.return_type.unify(other.return_type, seen=seen)
        for x, y in zip(self.params, other.params):
            can_unify &= x.type.unify(y.type, seen=seen)
        if not can_unify:
            return False

        # If one side has fewer params (and params_known is not True), then
        # extend its param list to match the other side
        if not self.params_known:
            self.is_variadic |= other.is_variadic
            self.params_known |= other.params_known
            while len(other.params) > len(self.params):
                self.params.append(other.params[len(self.params)])
        if not other.params_known:
            other.is_variadic |= self.is_variadic
            other.params_known |= self.params_known
            while len(self.params) > len(other.params):
                other.params.append(self.params[len(other.params)])

        # If any parameter names are missing, try to fill them in
        for x, y in zip(self.params, other.params):
            if not x.name and y.name:
                x.name = y.name
            elif not y.name and x.name:
                y.name = x.name

        return True

    def unify_with_args(self, concrete: "FunctionSignature") -> bool:
        """
        Unify a function's signature with a list of argument types.
        This is more flexible than unify() and is intended to check
        the function's type at a specific callsite.

        This function is not symmetric; `self` represents the prototype
        (e.g. with variadic args), whereas `concrete` represents the
        set of arguments at the callsite.
        """
        if len(self.params) > len(concrete.params):
            return False
        if not self.is_variadic and len(self.params) != len(concrete.params):
            return False
        can_unify = self.return_type.unify(concrete.return_type)
        for x, y in zip(self.params, concrete.params):
            can_unify &= x.type.unify(y.type)
        return can_unify


@dataclass(eq=False)
class StructDeclaration:
    """Representation of a C struct or union"""

    @dataclass(eq=False)
    class StructField:
        type: Type
        offset: int
        name: str

    size: int
    align: int
    tag_name: Optional[str]
    typedef_name: Optional[str]
    fields: List[StructField]  # sorted by `.offset`
    has_bitfields: bool
    is_union: bool

    def unify(
        self,
        other: "StructDeclaration",
        *,
        seen: Optional[Set["TypeData"]] = None,
    ) -> bool:
        # NB: Currently, the only structs that exist are defined from ctypes in the typemap,
        # so for now we can use reference equality to check if two structs are compatible.
        return self is other

    def fields_at_offset(self, offset: int) -> List[StructField]:
        """Return the list of StructFields which contain the given offset (in bits)"""
        fields = []
        for field in self.fields:
            # We assume fields are sorted by `offset`, ascending
            if field.offset > offset:
                break
            field_size = field.type.get_size_bytes()
            assert field_size is not None
            if field.offset + field_size < offset:
                continue
            fields.append(field)
        return fields

    @staticmethod
    def from_ctype(
        ctype: CStructUnion, typemap: TypeMap, typepool: TypePool
    ) -> "StructDeclaration":
        """
        Return StructDeclaration for a given ctype struct or union, constructing it
        and registering it in the typepool if it does not already exist.
        """
        existing_struct = typepool.get_struct_for_ctype(ctype)
        if existing_struct:
            return existing_struct

        struct = parse_struct(ctype, typemap)

        typedef_name: Optional[str] = None
        if ctype in typemap.struct_typedefs:
            typedef = typemap.struct_typedefs[ctype]
            assert isinstance(typedef.type, ca.IdentifierType)
            typedef_name = typedef.type.names[0]
        elif ctype.name and ctype.name in typemap.struct_typedefs:
            typedef = typemap.struct_typedefs[ctype.name]
            assert isinstance(typedef.type, ca.IdentifierType)
            typedef_name = typedef.type.names[0]

        assert (
            struct.size % struct.align == 0
        ), "struct size must be a multiple of its alignment"

        decl = StructDeclaration(
            size=struct.size,
            align=struct.align,
            tag_name=ctype.name,
            typedef_name=typedef_name,
            fields=[],
            has_bitfields=struct.has_bitfields,
            is_union=isinstance(ctype, ca.Union),
        )
        # Register the struct in the typepool now, before parsing the fields,
        # in case there are any self-referential fields in this struct.
        typepool.add_struct(decl, ctype)

        for offset, fields in sorted(struct.fields.items()):
            for field in fields:
                field_type = Type.ctype(field.type, typemap, typepool)
                assert field.size == field_type.get_size_bytes(), (
                    field.size,
                    field_type.get_size_bytes(),
                    field.name,
                    field_type,
                )
                decl.fields.append(
                    StructDeclaration.StructField(
                        type=field_type,
                        offset=offset,
                        name=field.name,
                    )
                )
        assert decl.fields == sorted(decl.fields, key=lambda f: f.offset)

        return decl
