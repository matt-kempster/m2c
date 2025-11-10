from __future__ import annotations

import unittest

from m2c.arch_x86 import X86Arch
from m2c.asm_instruction import (
    AsmAddressMode,
    AsmLiteral,
    AsmState,
    RegFormatter,
    parse_asm_instruction,
)
from m2c.asm_instruction import Register


class TestX86Parsing(unittest.TestCase):
    def setUp(self) -> None:
        self.arch = X86Arch()
        self.asm_state = AsmState(reg_formatter=RegFormatter())

    def test_dword_ptr_address_mode(self) -> None:
        instr = parse_asm_instruction(
            "mov eax, dword ptr [esp + 0xc]", self.arch, self.asm_state
        )
        # Size prefixes should be discarded so we only have two arguments.
        self.assertEqual(len(instr.args), 2)
        addr = instr.args[1]
        self.assertIsInstance(addr, AsmAddressMode)
        self.assertEqual(addr.base, Register("esp"))
        self.assertEqual(addr.addend, AsmLiteral(0xC))

    def test_absolute_symbol_address_mode(self) -> None:
        instr = parse_asm_instruction(
            "mov eax, [_DAT_0079a8b0]", self.arch, self.asm_state
        )
        self.assertEqual(len(instr.args), 2)
        addr = instr.args[1]
        self.assertIsInstance(addr, AsmAddressMode)
        self.assertEqual(addr.base, Register("zero"))
        self.assertEqual(addr.addend.symbol_name, "_DAT_0079a8b0")


if __name__ == "__main__":
    unittest.main()
