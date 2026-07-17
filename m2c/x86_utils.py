"""Small x86 asm-level helpers shared by the arch (m2c/arch_x86.py) and its
whole-body prepasses (the ESP-delta stack rewrite there, and the x87 FPU
rewrite in m2c/x86_fpu.py). They live in this leaf module so that x86_fpu.py
does not need to import from arch_x86.py, which imports x86_fpu.py."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from .asm_file import AsmData
from .asm_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmLiteral,
    BinOp,
    Register,
    ZERO,
    traverse_arg,
)
from .instruction import Instruction


# Operand-width mnemonic suffixes: parsing/normalization canonicalizes both
# "<size> ptr" memory prefixes and sub-register operand names into these
# ARM-style suffixes (see the arch_x86 module docstring). The default 32-bit
# width has no suffix.
WIDTH_SUFFIXES: Dict[int, str] = {1: ".b", 2: ".w", 4: "", 8: ".q"}


def split_width_suffix(mnemonic: str) -> Tuple[str, int]:
    """Split e.g. "mov.b" into ("mov", 1). No suffix means 4 bytes."""
    for width, suffix in WIDTH_SUFFIXES.items():
        if suffix and mnemonic.endswith(suffix):
            return mnemonic[: -len(suffix)], width
    return mnemonic, 4


def call_target_symbol(target: Argument) -> Optional[str]:
    """The symbol a call goes through: the name of a direct call target, or
    the absolute import slot of a `call [__imp__X]`-style indirect call."""
    if isinstance(target, AsmGlobalSymbol):
        return target.symbol_name
    if (
        isinstance(target, AsmAddressMode)
        and target.base == ZERO
        and isinstance(target.addend, AsmGlobalSymbol)
    ):
        return target.addend.symbol_name
    return None


def parse_jump_table_address(
    target: AsmAddressMode,
) -> Optional[Union[AsmGlobalSymbol, AsmLiteral]]:
    """Parse the table term from an x86 indirect jump address.

    Symbolic table terms are accepted for the normal switch path. A raw
    address is recognized only alongside a pointer-sized scaled index, which
    avoids classifying absolute indirect tail calls as switches.
    """
    symbols = [
        item
        for item in traverse_arg(target.addend)
        if isinstance(item, AsmGlobalSymbol)
    ]
    if len(symbols) == 1:
        return symbols[0]

    has_scaled_index = any(
        isinstance(item, BinOp)
        and item.op == "*"
        and isinstance(item.lhs, Register)
        and isinstance(item.rhs, AsmLiteral)
        and item.rhs.value == 4
        for item in traverse_arg(target.addend)
    )
    if not has_scaled_index:
        return None
    raw_addresses = [
        item
        for item in traverse_arg(target.addend)
        if isinstance(item, AsmLiteral) and item.value > 0xFFFF
    ]
    return raw_addresses[0] if len(raw_addresses) == 1 else None


def switch_jump_table_labels(
    instr: Instruction, asm_data: AsmData
) -> Optional[List[str]]:
    """For an indirect `jmp [index*4 + table]`, find the jump table in the
    file's data and return the list of case labels."""
    table = instr.jump_table
    if not isinstance(table, AsmGlobalSymbol):
        return None
    entry = asm_data.values.get(table.symbol_name)
    if entry is None or not entry.data:
        return None
    targets = []
    for item in entry.data:
        if isinstance(item, bytes):
            break
        target = item.as_symbol_without_addend()
        if target is None:
            break
        targets.append(target)
    if targets:
        return targets
    return None
