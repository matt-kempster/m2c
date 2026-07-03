from __future__ import annotations

import io
import unittest

from m2c.arch_mips import MipsArch
from m2c.arch_x86 import X86Arch
from m2c.asm_file import AsmFile, parse_file
from m2c.main import parse_flags


def _parse(source: str, target: str, arch: object) -> AsmFile:
    options = parse_flags(["--target", target, "irrelevant.s"])
    f = io.StringIO(source)
    f.name = "test.s"
    return parse_file(f, arch, options)  # type: ignore[arg-type]


class TestCrossFunctionMerge(unittest.TestCase):
    """merge_functions_with_cross_jumps rejoins Ghidra-style x86 functions that
    a mid-function global label split apart. It is gated to x86 (M2): other
    architectures may legitimately branch conditionally to another symbol, and
    such functions must not be silently fused."""

    # Two MIPS functions where the first conditionally branches to the second's
    # entry. A compiler would not emit this, but hand-authored asm / linker
    # labels can; it must stay two functions.
    MIPS_ASM = """
.set noat
.set noreorder

glabel func_a
/* 00 00 10000000 */  beq   $a0, $zero, func_b
/* 04 04 00000000 */   nop
/* 08 08 03E00008 */  jr    $ra
/* 0C 0C 00000000 */   nop

glabel func_b
/* 10 10 03E00008 */  jr    $ra
/* 14 14 00000000 */   nop
"""

    def test_mips_cross_symbol_branch_not_merged(self) -> None:
        asm_file = _parse(self.MIPS_ASM, "mips-ido-c", MipsArch())
        names = [fn.name for fn in asm_file.functions]
        self.assertEqual(names, ["func_a", "func_b"])

    # The equivalent x86 pattern (a conditional branch to another parsed
    # function's entry) IS merged: Ghidra split one function at a named label.
    X86_ASM = """
func_a:
    TEST EAX, EAX
    JZ func_b
    RET
func_b:
    MOV EAX, 0x1
    RET
"""

    def test_x86_cross_symbol_branch_merged(self) -> None:
        asm_file = _parse(self.X86_ASM, "x86-gcc-c", X86Arch())
        names = [fn.name for fn in asm_file.functions]
        self.assertEqual(names, ["func_a"])


class TestSetDirectiveLeniency(unittest.TestCase):
    """Non-integer `.set` values are warned-and-ignored only for x86/Ghidra
    inputs; other architectures keep the strict parse failure (L1)."""

    SET_ASM = """
.set noat
.set noreorder
.set some_symbol, table_base + 3

glabel func_a
/* 00 00 03E00008 */  jr    $ra
/* 04 04 00000000 */   nop
"""

    def test_non_x86_non_integer_set_raises(self) -> None:
        from m2c.error import DecompFailure

        with self.assertRaises(DecompFailure):
            _parse(self.SET_ASM, "mips-ido-c", MipsArch())

    X86_SET_ASM = """
.set some_symbol, table_base + 3
func_a:
    RET
"""

    def test_x86_non_integer_set_ignored(self) -> None:
        # x86 tolerates the non-integer .set (Ghidra emits label-equate
        # expressions) instead of failing.
        asm_file = _parse(self.X86_SET_ASM, "x86-gcc-c", X86Arch())
        self.assertEqual([fn.name for fn in asm_file.functions], ["func_a"])


if __name__ == "__main__":
    unittest.main()
