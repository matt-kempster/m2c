from typing import Optional, Set, Tuple, Union

import attr
import pycparser.c_ast as ca

from .c_types import Function as CFunction
from .c_types import (
    CType,
    TypeMap,
    equal_types,
    get_struct,
    parse_function,
    primitive_size,
    parse_struct,
    resolve_typedefs,
    type_to_string,
    var_size_align,
)


@attr.s(eq=False, repr=False)
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

    K_INT = 1
    K_PTR = 2
    K_FLOAT = 4
    K_CTYPE = 8
    K_INTPTR = K_INT | K_PTR
    K_ANYREG = K_INT | K_PTR | K_FLOAT
    K_ANY = K_INT | K_PTR | K_FLOAT | K_CTYPE

    SIGNED = 1
    UNSIGNED = 2
    ANY_SIGN = 3

    kind: int = attr.ib(default=K_ANY)
    size: Optional[int] = attr.ib(default=None)
    uf_parent: Optional["Type"] = attr.ib(default=None)

    sign: int = attr.ib(default=ANY_SIGN)  # K_INT
    ptr_to: Optional["Type"] = attr.ib(default=None)  # K_PTR
    typemap: Optional[TypeMap] = attr.ib(default=None)  # K_CTYPE
    ctype_ref: Optional[CType] = attr.ib(default=None)  # K_CTYPE

    def unify(self, other: "Type") -> bool:
        """
        Try to set this type equal to another. Returns true on success.
        Once set equal, the types will always be equal (we use a union-find
        structure to ensure this).
        """
        x = self.get_representative()
        y = other.get_representative()
        if x is y:
            return True
        if x.size is not None and y.size is not None and x.size != y.size:
            return False

        kind = x.kind & y.kind
        size = x.size if x.size is not None else y.size
        typemap = x.typemap if x.typemap is not None else y.typemap
        ctype_ref = x.ctype_ref if x.ctype_ref is not None else y.ctype_ref
        ptr_to = x.ptr_to if x.ptr_to is not None else y.ptr_to
        sign = x.sign & y.sign
        if size not in (None, 32, 64):
            kind &= ~Type.K_FLOAT
        if size not in (None, 32):
            kind &= ~Type.K_PTR
        if kind == 0 or sign == 0:
            return False
        if kind == Type.K_PTR:
            size = 32
        if sign != Type.ANY_SIGN:
            assert kind == Type.K_INT
        if x.ctype_ref is not None and y.ctype_ref is not None:
            assert typemap is not None
            x_ctype = resolve_typedefs(x.ctype_ref, typemap)
            y_ctype = resolve_typedefs(y.ctype_ref, typemap)
            if not equal_types(x_ctype, y_ctype):
                return False
        if x.ptr_to is not None and y.ptr_to is not None:
            if not x.ptr_to.unify(y.ptr_to):
                return False
        x.kind = kind
        x.size = size
        x.sign = sign
        x.ptr_to = ptr_to
        x.typemap = typemap
        x.ctype_ref = ctype_ref
        y.uf_parent = x
        return True

    def get_representative(self) -> "Type":
        if self.uf_parent is None:
            return self
        self.uf_parent = self.uf_parent.get_representative()
        return self.uf_parent

    def is_float(self) -> bool:
        return self.get_representative().kind == Type.K_FLOAT

    def is_pointer(self) -> bool:
        return self.get_representative().kind == Type.K_PTR

    def is_int(self) -> bool:
        return self.get_representative().kind == Type.K_INT

    def is_ctype(self) -> bool:
        return self.get_representative().kind == Type.K_CTYPE

    def is_unsigned(self) -> bool:
        return self.get_representative().sign == Type.UNSIGNED

    def get_size_bits(self) -> int:
        return self.get_representative().size or 32

    def get_pointer_to_ctype(self) -> Optional[CType]:
        """If self is a pointer-to-a-CType, return the CType"""
        type = self.get_representative()
        if type.is_pointer() and type.ptr_to is not None:
            ptr_to = type.ptr_to.get_representative()
            if ptr_to.is_ctype() and ptr_to.ctype_ref:
                return ptr_to.ctype_ref
        return None

    def parse_function(self) -> Optional[CFunction]:
        ctype = self.get_pointer_to_ctype()
        if ctype is not None:
            return parse_function(ctype)
        return None

    def to_decl(self, var: str) -> str:
        ret = str(self)
        prefix = ret if ret.endswith("*") else ret + " "
        return prefix + var

    def _stringify(self, seen: Set["Type"]) -> str:
        if self in seen:
            return "?"
        seen.add(self)
        type = self.get_representative()
        size = type.size or 32
        sign = "s" if type.sign & Type.SIGNED else "u"
        if type.kind in (Type.K_ANY, Type.K_ANYREG):
            if type.size is not None:
                return f"?{size}"
            return "?"
        if type.kind == Type.K_PTR:
            if type.ptr_to is None:
                return "void *"
            ctype = type.get_pointer_to_ctype()
            if ctype is not None:
                return type_to_string(ca.PtrDecl(quals=[], type=ctype))
            return (type.ptr_to._stringify(seen) + " *").replace("* *", "**")
        if type.kind == Type.K_FLOAT:
            return f"f{size}"
        if type.kind == Type.K_CTYPE:
            if type.ctype_ref is None:
                return "?"
            return type_to_string(type.ctype_ref)
        return f"{sign}{size}"

    def __str__(self) -> str:
        return self._stringify(set())

    def __repr__(self) -> str:
        type = self.get_representative()
        signstr = ("+" if type.sign & Type.SIGNED else "") + (
            "-" if type.sign & Type.UNSIGNED else ""
        )
        kindstr = (
            ("I" if type.kind & Type.K_INT else "")
            + ("P" if type.kind & Type.K_PTR else "")
            + ("F" if type.kind & Type.K_FLOAT else "")
        )
        sizestr = str(type.size) if type.size is not None else "?"
        return f"Type({signstr + kindstr + sizestr})"

    @staticmethod
    def any() -> "Type":
        return Type()

    @staticmethod
    def any_reg() -> "Type":
        return Type(kind=Type.K_ANYREG)

    @staticmethod
    def intish() -> "Type":
        return Type(kind=Type.K_INT)

    @staticmethod
    def intptr() -> "Type":
        return Type(kind=Type.K_INTPTR)

    @staticmethod
    def intptr32() -> "Type":
        return Type(kind=Type.K_INTPTR, size=32)

    @staticmethod
    def ptr(type: Optional["Type"] = None) -> "Type":
        return Type(kind=Type.K_PTR, size=32, ptr_to=type)

    @staticmethod
    def _ctype(ctype: CType, typemap: TypeMap, size: Optional[int]) -> "Type":
        return Type(kind=Type.K_CTYPE, size=size, ctype_ref=ctype, typemap=typemap)

    @staticmethod
    def f32() -> "Type":
        return Type(kind=Type.K_FLOAT, size=32)

    @staticmethod
    def floatish() -> "Type":
        return Type(kind=Type.K_FLOAT)

    @staticmethod
    def f64() -> "Type":
        return Type(kind=Type.K_FLOAT, size=64)

    @staticmethod
    def s8() -> "Type":
        return Type(kind=Type.K_INT, size=8, sign=Type.SIGNED)

    @staticmethod
    def u8() -> "Type":
        return Type(kind=Type.K_INT, size=8, sign=Type.UNSIGNED)

    @staticmethod
    def s16() -> "Type":
        return Type(kind=Type.K_INT, size=16, sign=Type.SIGNED)

    @staticmethod
    def u16() -> "Type":
        return Type(kind=Type.K_INT, size=16, sign=Type.UNSIGNED)

    @staticmethod
    def s32() -> "Type":
        return Type(kind=Type.K_INT, size=32, sign=Type.SIGNED)

    @staticmethod
    def u32() -> "Type":
        return Type(kind=Type.K_INT, size=32, sign=Type.UNSIGNED)

    @staticmethod
    def s64() -> "Type":
        return Type(kind=Type.K_INT, size=64, sign=Type.SIGNED)

    @staticmethod
    def u64() -> "Type":
        return Type(kind=Type.K_INT, size=64, sign=Type.UNSIGNED)

    @staticmethod
    def int64() -> "Type":
        return Type(kind=Type.K_INT, size=64)

    @staticmethod
    def of_size(size: int) -> "Type":
        return Type(kind=Type.K_ANY, size=size)

    @staticmethod
    def bool() -> "Type":
        return Type.intish()


def type_from_ctype(ctype: CType, typemap: TypeMap) -> Type:
    real_ctype = resolve_typedefs(ctype, typemap)
    if isinstance(real_ctype, (ca.PtrDecl, ca.ArrayDecl)):
        return Type.ptr(type_from_ctype(real_ctype.type, typemap))
    if isinstance(real_ctype, ca.FuncDecl):
        return Type._ctype(real_ctype, typemap, size=None)
    if isinstance(real_ctype, ca.TypeDecl):
        if isinstance(real_ctype.type, (ca.Struct, ca.Union)):
            struct = parse_struct(real_ctype.type, typemap)
            return Type._ctype(ctype, typemap, size=struct.size * 8)
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
        sign = Type.UNSIGNED if "unsigned" in names else Type.SIGNED
        return Type(kind=Type.K_INT, size=size, sign=sign, typemap=typemap)


def ptr_type_from_ctype(ctype: CType, typemap: TypeMap) -> Tuple[Type, bool]:
    real_ctype = resolve_typedefs(ctype, typemap)
    if isinstance(real_ctype, ca.ArrayDecl):
        return Type.ptr(type_from_ctype(real_ctype.type, typemap)), True
    if isinstance(real_ctype, ca.FuncDecl):
        return Type.ptr(Type._ctype(ctype, typemap, size=None)), True
    return Type.ptr(type_from_ctype(ctype, typemap)), False


def get_field(
    type: Type, offset: int, typemap: TypeMap, *, target_size: Optional[int]
) -> Tuple[Optional[str], Type, Type, bool]:
    """Returns field name, target type, target pointer type, and whether the field is an array."""
    if target_size is None and offset == 0:
        # We might as well take a pointer to the whole struct
        target = get_pointer_target(type, typemap)
        target_type = target[1] if target else Type.any()
        return None, target_type, type, False
    ctype = type.get_pointer_to_ctype()
    if ctype is None:
        return None, Type.any(), Type.ptr(), False
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
    return None, Type.any(), Type.ptr(), False


def find_substruct_array(
    type: Type, offset: int, scale: int, typemap: TypeMap
) -> Optional[Tuple[str, int, Type]]:
    if scale <= 0:
        return None
    ctype = type.get_pointer_to_ctype()
    if ctype is None:
        return None
    ctype = resolve_typedefs(ctype, typemap)
    if not isinstance(ctype, ca.TypeDecl):
        return None
    if not isinstance(ctype.type, (ca.Struct, ca.Union)):
        return None
    struct = get_struct(ctype.type, typemap)
    if not struct:
        return None
    for off, fields in struct.fields.items():
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


def get_pointer_target(
    type: Type, typemap: Optional[TypeMap]
) -> Optional[Tuple[int, Type]]:
    type = type.get_representative()
    target = type.ptr_to
    if target is None:
        return None
    target = target.get_representative()
    ctype = type.get_pointer_to_ctype()
    if not ctype:
        if target.size is None:
            return None
        return target.size // 8, target
    if typemap is None:
        # (shouldn't happen, but might as well handle it)
        return None
    size, align = var_size_align(ctype, typemap)
    if align == 0:
        # void* or function pointer
        return None
    return size, target
