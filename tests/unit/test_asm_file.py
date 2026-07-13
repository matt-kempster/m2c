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


class TestSetDirectiveLeniency(unittest.TestCase):
    """Non-integer `.set` values (label-equate expressions emitted by
    disassemblers) are ignored on every architecture."""

    SET_ASM = """
.set noat
.set noreorder
.set some_symbol, table_base + 3

glabel func_a
/* 00 00 03E00008 */  jr    $ra
/* 04 04 00000000 */   nop
"""

    def test_mips_non_integer_set_ignored(self) -> None:
        asm_file = _parse(self.SET_ASM, "mips-ido-c", MipsArch())
        self.assertEqual([fn.name for fn in asm_file.functions], ["func_a"])

    X86_SET_ASM = """
.set some_symbol, table_base + 3
func_a:
    RET
"""

    def test_x86_non_integer_set_ignored(self) -> None:
        asm_file = _parse(self.X86_SET_ASM, "x86-gcc-c", X86Arch())
        self.assertEqual([fn.name for fn in asm_file.functions], ["func_a"])


class TestAddressModeTraversal(unittest.TestCase):
    """The shared AsmAddressMode gained an Optional base register for x86's
    base-less operands. A non-x86 address mode (which always has a base) must
    still traverse that base -- e.g. a MIPS load reads its base register."""

    def test_mips_load_base_register_is_input(self) -> None:
        from m2c.asm_instruction import (
            AsmAddressMode,
            AsmState,
            RegFormatter,
            Register,
            parse_asm_instruction,
            traverse_arg,
        )
        from m2c.instruction import InstructionMeta

        arch = MipsArch()
        state = AsmState(reg_formatter=RegFormatter())
        asm = parse_asm_instruction("lw $t0, 4($a0)", arch, state)
        mem = asm.args[1]
        assert isinstance(mem, AsmAddressMode)
        self.assertEqual(mem.base, Register("a0"))
        # The shared traversal yields the (non-None) base register.
        self.assertIn(Register("a0"), list(traverse_arg(mem)))
        instr = arch.parse(asm.mnemonic, asm.args, InstructionMeta.missing())
        self.assertIn(Register("a0"), instr.inputs)


if __name__ == "__main__":
    unittest.main()
