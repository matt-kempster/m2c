from typing import List, Optional, Union

import attr

from .options import Formatter
from .types import Type
from .translate import (
    as_type,
    GlobalInfo,
    GlobalSymbol,
    Expression,
    Initializer,
    Literal,
)


@attr.s
class GenericInitializer(Initializer):
    global_info: GlobalInfo = attr.ib()
    fmt: Formatter = attr.ib()
    # Data currently being formatted
    data: List[Union[str, bytes]] = attr.ib(factory=list)

    def for_symbol(self, sym: GlobalSymbol) -> Optional[str]:
        """Generate the C initializer expression for sym"""
        if not sym.asm_data_entry:
            return None
        self.data = sym.asm_data_entry.data[:]

        if sym.array_dim is None:
            return self.for_type(sym.type)
        else:
            elements: List[str] = []
            for _ in range(sym.array_dim):
                el = self.for_type(sym.type)
                if el is None:
                    return None
                elements.append(el)
            return f"{{{', '.join(elements)}}}"

    def for_type(self, type: Type) -> Optional[str]:
        if type.is_int() or type.is_float():
            size_bits = type.get_size_bits()
            if size_bits == 0:
                return None
            elif size_bits is None:
                # Unknown size; guess 32 bits
                size_bits = 32
            value = self.read_uint(size_bits // 8)
            if value is not None:
                return Literal(value, type).format(self.fmt)

        if type.is_pointer():
            ptr = self.read_pointer()
            if ptr is not None:
                return as_type(ptr, type, True).format(self.fmt)

        if type.is_ctype():
            # TODO: Generate initializers for structs/arrays/etc.
            return None

        # Type kinds K_FN and K_VOID do not have initializers
        return None

    def read_uint(self, n: int) -> Optional[int]:
        """Read the next `n` bytes from `data` as an (long) integer"""
        assert 0 < n <= 8
        if not self.data or not isinstance(self.data[0], bytes):
            return None
        if len(self.data[0]) < n:
            return None
        bs = self.data[0][:n]
        self.data[0] = self.data[0][n:]
        if not self.data[0]:
            del self.data[0]
        value = 0
        for b in bs:
            value = (value << 8) | b
        return value

    def read_pointer(self) -> Optional[Expression]:
        """Read the next label from `data`"""
        if not self.data:
            return None

        if not isinstance(self.data[0], str):
            # Bare pointer
            value = self.read_uint(4)
            if value is None:
                return None
            return Literal(value=value)

        # Pointer label
        label = self.data.pop(0)
        assert isinstance(label, str)
        return self.global_info.address_of_gsym(label)
