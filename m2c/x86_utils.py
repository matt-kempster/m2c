"""Small x86 asm-level helpers shared by the arch (m2c/arch_x86.py) and its
whole-body prepasses (the ESP-delta stack rewrite there, and the x87 FPU
rewrite in m2c/x86_fpu.py). They live in this leaf module so that x86_fpu.py
does not need to import from arch_x86.py, which imports x86_fpu.py."""

from __future__ import annotations

from typing import List, Optional, Tuple

from .asm_file import AsmData
from .asm_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    ZERO,
    traverse_arg,
)
from .instruction import Instruction
from .intel_syntax import WIDTH_SUFFIXES


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


def switch_jump_table_labels(
    instr: Instruction, asm_data: AsmData
) -> Optional[List[str]]:
    """For an indirect `jmp [index*4 + table]`, find the jump table in the
    file's data and return the list of case labels."""
    for arg in instr.args:
        if not isinstance(arg, AsmAddressMode):
            continue
        for sub in traverse_arg(arg.addend):
            if not isinstance(sub, AsmGlobalSymbol):
                continue
            entry = asm_data.values.get(sub.symbol_name)
            if entry is None or not entry.data:
                continue
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
