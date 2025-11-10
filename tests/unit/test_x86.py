from __future__ import annotations

import unittest

from m2c.arch_x86 import X86Arch
from m2c.asm_instruction import (
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmLiteral,
    AsmState,
    RegFormatter,
    Register,
    parse_asm_instruction,
)
from m2c.instruction import Instruction, InstructionMeta, StackLocation


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

    def parse_instruction(self, line: str) -> Instruction:
        asm = parse_asm_instruction(line, self.arch, AsmState(reg_formatter=RegFormatter()))
        return self.arch.parse(asm.mnemonic, asm.args, InstructionMeta.missing())

    def test_mov_stack_load_instruction(self) -> None:
        instr = self.parse_instruction("mov eax, dword ptr [esp + 0x4]")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertTrue(instr.inputs, "mov stack load should read memory")
        loc = instr.inputs[0]
        self.assertIsInstance(loc, StackLocation)
        self.assertEqual(loc.offset, 4)

    def test_mov_absolute_load_instruction(self) -> None:
        instr = self.parse_instruction("mov ecx, [_DAT_00401000]")
        self.assertEqual(instr.outputs, [Register("ecx")])
        self.assertTrue(instr.eval_fn, "mov instruction should have eval_fn")

    def test_push_register_updates_stack(self) -> None:
        instr = self.parse_instruction("push eax")
        self.assertEqual(instr.outputs, [Register("esp")])
        self.assertIn(Register("esp"), instr.inputs)

    def test_sub_esp_allocates_stack(self) -> None:
        instr = self.parse_instruction("sub esp, 0x8")
        self.assertEqual(instr.outputs, [Register("esp")])
        self.assertIn(Register("esp"), instr.inputs)

    def test_test_sets_flags(self) -> None:
        instr = self.parse_instruction("test eax, eax")
        self.assertIn(Register("eax"), instr.inputs)

    def test_xor_zeroes_register(self) -> None:
        instr = self.parse_instruction("xor eax, eax")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("eax"), instr.inputs)

    def test_call_symbol(self) -> None:
        instr = self.parse_instruction("call _malloc")
        self.assertEqual(instr.function_target, AsmGlobalSymbol("_malloc"))
        self.assertEqual(instr.outputs, [Register("eax"), Register("edx")])

    def test_mov_stack_store_instruction(self) -> None:
        instr = self.parse_instruction("mov dword ptr [esp + 0x8], eax")
        self.assertTrue(instr.is_store)
        self.assertIn(Register("eax"), instr.inputs)
        loc = instr.outputs[0]
        self.assertIsInstance(loc, StackLocation)
        self.assertEqual(loc.offset, 8)

    def test_mov_base_offset_store_instruction(self) -> None:
        instr = self.parse_instruction("mov dword ptr [eax + 0x1c], ecx")
        self.assertTrue(instr.is_store)
        self.assertIn(Register("ecx"), instr.inputs)
        self.assertIn(Register("eax"), instr.inputs)


if __name__ == "__main__":
    unittest.main()
