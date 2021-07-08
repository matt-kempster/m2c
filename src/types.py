import copy
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Union

import pycparser.c_ast as ca

from .c_types import (
    CType,
    Struct,
    TypeMap,
    equal_types,
    get_struct,
    parse_constant_int,
    parse_function,
    parse_struct,
    primitive_size,
    resolve_typedefs,
    set_decl_name,
    to_c,
    var_size_align,
)
from .error import DecompFailure
from .options import Formatter


@dataclass(eq=False)
class TypeData:
    K_INT = 1
    K_PTR = 2
    K_FLOAT = 4
    K_CTYPE = 8
    K_FN = 16
    K_VOID = 32
    K_INTPTR = K_INT | K_PTR
    K_ANYREG = K_INT | K_PTR | K_FLOAT
    K_ANY = K_INT | K_PTR | K_FLOAT | K_CTYPE | K_FN | K_VOID

    SIGNED = 1
    UNSIGNED = 2
    ANY_SIGN = 3

    kind: int = K_ANY
    likely_kind: int = K_ANY  # subset of kind
    size: Optional[int] = None
    uf_parent: Optional["TypeData"] = None

    sign: int = ANY_SIGN  # K_INT
    ptr_to: Optional["Type"] = None  # K_PTR
    typemap: Optional[TypeMap] = None  # K_CTYPE
    ctype_ref: Optional[CType] = None  # K_CTYPE
    fn_sig: Optional["FunctionSignature"] = None  # K_FN

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

        if x.size is not None and y.size is not None and x.size != y.size:
            return False

        kind = x.kind & y.kind
        likely_kind = x.likely_kind & y.likely_kind
        size = x.size if x.size is not None else y.size
        typemap = x.typemap if x.typemap is not None else y.typemap
        ctype_ref = x.ctype_ref if x.ctype_ref is not None else y.ctype_ref
        ptr_to = x.ptr_to if x.ptr_to is not None else y.ptr_to
        fn_sig = x.fn_sig if x.fn_sig is not None else y.fn_sig
        sign = x.sign & y.sign
        if size not in (None, 32, 64):
            kind &= ~TypeData.K_FLOAT
        if size not in (None, 32):
            kind &= ~TypeData.K_PTR
        if size not in (None,):
            kind &= ~TypeData.K_FN
        if size not in (None, 0):
            kind &= ~TypeData.K_VOID
        likely_kind &= kind
        if kind == 0 or sign == 0:
            return False
        if kind == TypeData.K_PTR:
            size = 32
        if sign != TypeData.ANY_SIGN:
            assert kind == TypeData.K_INT
        if x.ctype_ref is not None and y.ctype_ref is not None:
            assert typemap is not None
            x_ctype = resolve_typedefs(x.ctype_ref, typemap)
            y_ctype = resolve_typedefs(y.ctype_ref, typemap)
            if not equal_types(x_ctype, y_ctype):
                return False
        if x.ptr_to is not None and y.ptr_to is not None:
            if not x.ptr_to.unify(y.ptr_to, seen=seen):
                return False
        if x.fn_sig is not None and y.fn_sig is not None:
            if not x.fn_sig.unify(y.fn_sig, seen=seen):
                return False
        x.kind = kind
        x.likely_kind = likely_kind
        x.size = size
        x.sign = sign
        x.ptr_to = ptr_to
        x.typemap = typemap
        x.ctype_ref = ctype_ref
        x.fn_sig = fn_sig
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

    def is_int(self) -> bool:
        return self.data().kind == TypeData.K_INT

    def is_reg(self) -> bool:
        return (self.data().kind & ~TypeData.K_ANYREG) == 0

    def is_ctype(self) -> bool:
        return self.data().kind == TypeData.K_CTYPE

    def is_function(self) -> bool:
        return self.data().kind == TypeData.K_FN

    def is_void(self) -> bool:
        return self.data().kind == TypeData.K_VOID

    def is_unsigned(self) -> bool:
        return self.data().sign == TypeData.UNSIGNED

    def get_size_bits(self) -> Optional[int]:
        return self.data().size

    def get_size_bytes(self) -> Optional[int]:
        size = self.get_size_bits()
        return None if size is None else size // 8

    def get_size_align_bytes(self) -> Tuple[int, int]:
        data = self.data()
        if data.kind == TypeData.K_CTYPE and data.ctype_ref is not None:
            assert data.typemap is not None
            return var_size_align(data.ctype_ref, data.typemap)
        size = (self.get_size_bits() or 32) // 8
        return size, size

    def is_struct_type(self) -> bool:
        data = self.data()
        if data.kind != TypeData.K_CTYPE or data.ctype_ref is None:
            return False
        assert data.typemap is not None
        ctype = resolve_typedefs(data.ctype_ref, data.typemap)
        if not isinstance(ctype, ca.TypeDecl):
            return False
        return isinstance(ctype.type, (ca.Struct, ca.Union))

    def get_pointer_target(self) -> Optional["Type"]:
        """If self is a pointer-to-a-Type, return the Type"""
        data = self.data()
        if self.is_pointer() and data.ptr_to is not None:
            return data.ptr_to
        return None

    def get_ctype_and_typemap(self) -> Optional[Tuple[CType, TypeMap]]:
        """If self is a CType, return the CType and the TypeMap it came from"""
        data = self.data()
        if data.kind == TypeData.K_CTYPE:
            assert data.ctype_ref is not None and data.typemap is not None
            return data.ctype_ref, data.typemap
        return None

    def get_function_pointer_signature(self) -> Optional["FunctionSignature"]:
        """If self is a function pointer, return the FunctionSignature"""
        data = self.data()
        if self.is_pointer() and data.ptr_to is not None:
            ptr_to = data.ptr_to.data()
            if ptr_to.kind == TypeData.K_FN:
                return ptr_to.fn_sig
        return None

    def get_ctype_fields(
        self,
    ) -> Optional[List[Union[int, "Type"]]]:
        """
        If self is a CType, get a list of fields, suitable for creating an initializer,
        or return None if an initializer cannot be made (e.g. a struct with bitfields)

        Struct padding is represented by an int in the list, otherwise the list members
        denote the field's Type.
        """
        data = self.data()
        if data.kind != TypeData.K_CTYPE or data.ctype_ref is None:
            return None
        assert data.typemap is not None
        ctype = resolve_typedefs(data.ctype_ref, data.typemap)

        # ArrayDecls are still used to when representing pointers-to-arrays, or
        # multidimensional arrays.
        # Treat an array of length N as a struct with N (identical) members
        if isinstance(ctype, ca.ArrayDecl):
            field_type, dim = array_type_and_dim(ctype, data.typemap)
            if not dim:
                # Do not support zero-sized arrays
                return None
            return [field_type] * dim

        # Lookup the c_types.Struct representation
        if not isinstance(ctype, ca.TypeDecl):
            return None
        if not isinstance(ctype.type, (ca.Struct, ca.Union)):
            return None
        struct = get_struct(ctype.type, data.typemap)
        if not struct or struct.has_bitfields:
            # Bitfields aren't supported; they aren't represented in `struct.fields`
            return None

        output: List[Union[int, Type]] = []
        position = 0
        for offset, fields in sorted(struct.fields.items()):
            if offset < position:
                # Overlapping fields, e.g. from expanded struct paths
                continue
            elif offset > position:
                # Padding bytes
                output.append(offset - position)

            # Choose the first field in a union, or the unexpanded name in a struct
            field = fields[0]
            field_type = type_from_ctype(field.type, data.typemap, array_decay=False)
            size = field_type.get_size_bytes()
            if not size:
                return None
            output.append(field_type)
            position = offset + size

        assert position <= struct.size
        if position < struct.size:
            # Insert padding bytes
            output.append(struct.size - position)

        return output

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
        size = data.size or 32
        sign = "s" if data.sign & TypeData.SIGNED else "u"

        if (data.kind & TypeData.K_ANYREG) == TypeData.K_ANYREG and (
            data.likely_kind & (TypeData.K_INT | TypeData.K_FLOAT)
        ) not in (TypeData.K_INT, TypeData.K_FLOAT):
            if data.size is not None:
                return simple_ctype(f"{unk_symbol}{size}")
            return simple_ctype(unk_symbol)

        if (
            data.kind == TypeData.K_FLOAT
            or (data.likely_kind & (TypeData.K_FLOAT | TypeData.K_INT))
            == TypeData.K_FLOAT
        ):
            return simple_ctype(f"f{size}")

        if data.kind == TypeData.K_PTR:
            if data.ptr_to is None:
                return ca.PtrDecl(type=simple_ctype("void"), quals=[])
            return ca.PtrDecl(type=data.ptr_to._to_ctype(seen, fmt), quals=[])

        if data.kind == TypeData.K_CTYPE:
            if data.ctype_ref is None:
                return simple_ctype(unk_symbol)
            ctype = copy.deepcopy(data.ctype_ref)
            if isinstance(ctype, ca.TypeDecl) and isinstance(
                ctype.type, (ca.Struct, ca.Union)
            ):
                if ctype.type.name is not None:
                    # Remove struct field declarations for named structs
                    ctype.type.decls = None
            return ctype

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

        return simple_ctype(f"{sign}{size}")

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
            + ("C" if data.kind & TypeData.K_CTYPE else "")
            + ("N" if data.kind & TypeData.K_FN else "")
            + ("V" if data.kind & TypeData.K_VOID else "")
        )
        sizestr = str(data.size) if data.size is not None else "?"
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
        return Type(TypeData(kind=TypeData.K_PTR, size=32, ptr_to=type))

    @staticmethod
    def _ctype(ctype: CType, typemap: TypeMap, size: Optional[int]) -> "Type":
        return Type(
            TypeData(kind=TypeData.K_CTYPE, size=size, ctype_ref=ctype, typemap=typemap)
        )

    @staticmethod
    def function(fn_sig: Optional["FunctionSignature"] = None) -> "Type":
        if fn_sig is None:
            fn_sig = FunctionSignature()
        return Type(TypeData(kind=TypeData.K_FN, fn_sig=fn_sig))

    @staticmethod
    def f32() -> "Type":
        return Type(TypeData(kind=TypeData.K_FLOAT, size=32))

    @staticmethod
    def floatish() -> "Type":
        return Type(TypeData(kind=TypeData.K_FLOAT))

    @staticmethod
    def f64() -> "Type":
        return Type(TypeData(kind=TypeData.K_FLOAT, size=64))

    @staticmethod
    def s8() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=8, sign=TypeData.SIGNED))

    @staticmethod
    def u8() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=8, sign=TypeData.UNSIGNED))

    @staticmethod
    def s16() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=16, sign=TypeData.SIGNED))

    @staticmethod
    def u16() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=16, sign=TypeData.UNSIGNED))

    @staticmethod
    def s32() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=32, sign=TypeData.SIGNED))

    @staticmethod
    def u32() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=32, sign=TypeData.UNSIGNED))

    @staticmethod
    def s64() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=64, sign=TypeData.SIGNED))

    @staticmethod
    def u64() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=64, sign=TypeData.UNSIGNED))

    @staticmethod
    def int64() -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=64))

    @staticmethod
    def int_of_size(size: int) -> "Type":
        return Type(TypeData(kind=TypeData.K_INT, size=size))

    @staticmethod
    def reg32(*, likely_float: bool) -> "Type":
        likely = TypeData.K_FLOAT if likely_float else TypeData.K_INTPTR
        return Type(TypeData(kind=TypeData.K_ANYREG, likely_kind=likely, size=32))

    @staticmethod
    def reg64(*, likely_float: bool) -> "Type":
        kind = TypeData.K_FLOAT | TypeData.K_INT
        likely = TypeData.K_FLOAT if likely_float else TypeData.K_INT
        return Type(TypeData(kind=kind, likely_kind=likely, size=64))

    @staticmethod
    def bool() -> "Type":
        return Type.intish()

    @staticmethod
    def void() -> "Type":
        return Type(TypeData(kind=TypeData.K_VOID, size=0))


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


def type_from_ctype(ctype: CType, typemap: TypeMap, array_decay: bool = True) -> Type:
    real_ctype = resolve_typedefs(ctype, typemap)
    if isinstance(real_ctype, ca.ArrayDecl):
        inner_type, dim = array_type_and_dim(real_ctype, typemap)
        if array_decay:
            return Type.ptr(inner_type)
        size = inner_type.get_size_bits()
        if size is not None and dim is not None:
            size *= dim
        else:
            size = None
        return Type._ctype(real_ctype, typemap, size=size)
    if isinstance(real_ctype, ca.PtrDecl):
        return Type.ptr(type_from_ctype(real_ctype.type, typemap, array_decay=False))
    if isinstance(real_ctype, ca.FuncDecl):
        fn = parse_function(real_ctype)
        assert fn is not None
        fn_sig = FunctionSignature(
            return_type=Type.void(),
            is_variadic=fn.is_variadic,
        )
        if fn.ret_type is not None:
            fn_sig.return_type = type_from_ctype(fn.ret_type, typemap)
        if fn.params is not None:
            fn_sig.params = [
                FunctionParam(
                    name=param.name or "",
                    type=type_from_ctype(param.type, typemap),
                )
                for param in fn.params
            ]
            fn_sig.params_known = True
        return Type.function(fn_sig)
    if isinstance(real_ctype, ca.TypeDecl):
        if isinstance(real_ctype.type, (ca.Struct, ca.Union)):
            struct = parse_struct(real_ctype.type, typemap)
            return Type._ctype(struct.type, typemap, size=struct.size * 8)
        names = (
            ["int"] if isinstance(real_ctype.type, ca.Enum) else real_ctype.type.names
        )
        if "double" in names:
            return Type.f64()
        if "float" in names:
            return Type.f32()
        size = 8 * primitive_size(real_ctype.type)
        if not size:
            return Type._ctype(ctype, typemap, size=None)
        sign = TypeData.UNSIGNED if "unsigned" in names else TypeData.SIGNED
        return Type(TypeData(kind=TypeData.K_INT, size=size, sign=sign))


def array_type_and_dim(ctype: ca.ArrayDecl, typemap: TypeMap) -> Tuple[Type, int]:
    dim = 0
    try:
        if ctype.dim is not None:
            dim = parse_constant_int(ctype.dim, typemap)
    except DecompFailure:
        pass
    return (
        type_from_ctype(ctype.type, typemap, array_decay=False),
        dim,
    )


def ptr_type_from_ctype(ctype: CType, typemap: TypeMap) -> Tuple[Type, Optional[int]]:
    real_ctype = resolve_typedefs(ctype, typemap)
    if isinstance(real_ctype, ca.ArrayDecl):
        # Array to pointer decay
        inner_type, dim = array_type_and_dim(real_ctype, typemap)
        return Type.ptr(inner_type), dim
    return Type.ptr(type_from_ctype(ctype, typemap)), None


def get_field(
    type: Type, offset: int, *, target_size: Optional[int]
) -> Tuple[Optional[str], Type, Type, Optional[int]]:
    """Returns field name, target type, target pointer type, and
    the field's array size (or None if the field is not an array)."""
    if target_size is None and offset == 0:
        # We might as well take a pointer to the whole struct
        target = type.get_pointer_target() or Type.any()
        return None, target, type, None

    deref_type = type.get_pointer_target()
    if deref_type is None:
        return None, Type.any(), Type.ptr(), None
    ctype_and_typemap = deref_type.get_ctype_and_typemap()
    if ctype_and_typemap is None:
        return None, Type.any(), Type.ptr(), None
    ctype, typemap = ctype_and_typemap
    ctype = resolve_typedefs(ctype, typemap)

    if isinstance(ctype, ca.TypeDecl) and isinstance(ctype.type, (ca.Struct, ca.Union)):
        struct = get_struct(ctype.type, typemap)
        if struct:
            fields = struct.fields.get(offset)
            if fields:
                # Ideally, we should use target_size and the target pointer type to
                # determine which struct field to use if there are multiple at the
                # same offset (e.g. if a struct starts here, or we have a union).
                # For now though, we just use target_size as a boolean signal -- if
                # it's known we take an arbitrary subfield that's as concrete as
                # possible, if unknown we prefer a whole substruct. (The latter case
                # happens when taking pointers to fields -- pointers to substructs are
                # more common and can later be converted to concrete field pointers.)
                if target_size is None:
                    # Structs will be placed first in the field list.
                    field = fields[0]
                else:
                    # Pick the first subfield in case of unions.
                    correct_size_fields = [f for f in fields if f.size == target_size]
                    if len(correct_size_fields) == 1:
                        field = correct_size_fields[0]
                    else:
                        ind = 0
                        while ind + 1 < len(fields) and fields[ind + 1].name.startswith(
                            fields[ind].name + "."
                        ):
                            ind += 1
                        field = fields[ind]
                return (
                    field.name,
                    type_from_ctype(field.type, typemap),
                    *ptr_type_from_ctype(field.type, typemap),
                )
    return None, Type.any(), Type.ptr(), None


def find_substruct_array(
    type: Type, offset: int, scale: int
) -> Optional[Tuple[str, int, Type]]:
    if scale <= 0:
        return None
    deref_type = type.get_pointer_target()
    if deref_type is None:
        return None
    ctype_and_typemap = deref_type.get_ctype_and_typemap()
    if ctype_and_typemap is None:
        return None
    ctype, typemap = ctype_and_typemap
    ctype = resolve_typedefs(ctype, typemap)
    if not isinstance(ctype, ca.TypeDecl):
        return None
    if not isinstance(ctype.type, (ca.Struct, ca.Union)):
        return None
    struct = get_struct(ctype.type, typemap)
    if not struct:
        return None
    for off, fields in sorted(struct.fields.items()):
        if offset < off:
            continue
        for field in fields:
            if offset >= off + field.size:
                continue
            field_type = resolve_typedefs(field.type, typemap)
            if not isinstance(field_type, ca.ArrayDecl):
                continue
            size = var_size_align(field_type.type, typemap)[0]
            if size == scale:
                return field.name, off, type_from_ctype(field_type.type, typemap)
    return None
