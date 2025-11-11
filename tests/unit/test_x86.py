from __future__ import annotations

import unittest

from typing import Tuple

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
        instr, _ = self.parse_instruction_with_state(line)
        return instr

    def parse_instruction_with_state(self, line: str) -> Tuple[Instruction, AsmState]:
        asm_state = AsmState(reg_formatter=RegFormatter())
        # Normalize register names to emulate how real input files spell them.
        asm = parse_asm_instruction(line.lower(), self.arch, asm_state)
        instr = self.arch.parse(asm.mnemonic, asm.args, InstructionMeta.missing())
        return instr, asm_state

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

    def test_pop_register_updates_stack(self) -> None:
        instr = self.parse_instruction("pop esi")
        self.assertIn(Register("esp"), instr.inputs)
        self.assertEqual(instr.outputs, [Register("esi"), Register("esp")])

    def test_sub_esp_allocates_stack(self) -> None:
        instr = self.parse_instruction("sub esp, 0x8")
        self.assertEqual(instr.outputs, [Register("esp")])
        self.assertIn(Register("esp"), instr.inputs)

    def test_test_sets_flags(self) -> None:
        instr = self.parse_instruction("test eax, eax")
        self.assertIn(Register("eax"), instr.inputs)

    def test_test_immediate(self) -> None:
        instr = self.parse_instruction("test eax, 0x1")
        self.assertIn(Register("eax"), instr.inputs)

    def test_test_stack_memory_immediate(self) -> None:
        instr = self.parse_instruction("test [esp + 0x8], 0x2")
        self.assertIn(Register("esp"), instr.inputs)
        self.assertEqual(instr.outputs, [Register("zf")])

    def test_xor_zeroes_register(self) -> None:
        instr = self.parse_instruction("xor eax, eax")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("eax"), instr.inputs)

    def test_call_symbol(self) -> None:
        instr = self.parse_instruction("call _malloc")
        self.assertEqual(instr.function_target, AsmGlobalSymbol("_malloc"))
        self.assertEqual(instr.outputs, [Register("eax"), Register("edx")])

    def test_push_offset_symbol(self) -> None:
        instr = self.parse_instruction("push offset _FUN_0040e440")
        self.assertEqual(instr.outputs, [Register("esp")])
        self.assertIn(Register("esp"), instr.inputs)

    def test_add_esp_releases_stack(self) -> None:
        instr = self.parse_instruction("add esp, 0x4")
        self.assertEqual(instr.outputs, [Register("esp")])
        self.assertIn(Register("esp"), instr.inputs)

    def test_add_register_immediate(self) -> None:
        instr = self.parse_instruction("add ecx, 0x3c")
        self.assertEqual(instr.outputs, [Register("ecx")])
        self.assertIn(Register("ecx"), instr.inputs)

    def test_add_register_register(self) -> None:
        instr = self.parse_instruction("add eax, edx")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertCountEqual(instr.inputs, [Register("eax"), Register("edx")])

    def test_sub_register_register(self) -> None:
        instr = self.parse_instruction("sub eax, ecx")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertCountEqual(instr.inputs, [Register("eax"), Register("ecx")])

    def test_shl_register_immediate(self) -> None:
        instr = self.parse_instruction("shl ecx, 0x8")
        self.assertEqual(instr.outputs, [Register("ecx")])
        self.assertIn(Register("ecx"), instr.inputs)

    def test_sar_register_immediate(self) -> None:
        instr = self.parse_instruction("sar eax, 0x8")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("eax"), instr.inputs)

    def test_shr_register_immediate(self) -> None:
        instr = self.parse_instruction("shr eax, 0x8")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("eax"), instr.inputs)

    def test_setz_register(self) -> None:
        instr = self.parse_instruction("setz eax")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertCountEqual(instr.inputs, [Register("zf")])

    def test_setnz_register(self) -> None:
        instr = self.parse_instruction("setnz eax")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertCountEqual(instr.inputs, [Register("zf")])

    def test_setge_register(self) -> None:
        instr = self.parse_instruction("setge eax")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertCountEqual(instr.inputs, [Register("sf"), Register("of")])

    def test_setg_register(self) -> None:
        instr = self.parse_instruction("setg eax")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertCountEqual(instr.inputs, [Register("zf"), Register("sf"), Register("of")])

    def test_setl_register(self) -> None:
        instr = self.parse_instruction("setl eax")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertCountEqual(instr.inputs, [Register("sf"), Register("of")])

    def test_cmp_reg_imm(self) -> None:
        instr = self.parse_instruction("cmp eax, 0x1b5")
        self.assertEqual(instr.outputs, [Register("zf")])
        self.assertIn(Register("eax"), instr.inputs)

    def test_cmp_reg_small_imm(self) -> None:
        instr = self.parse_instruction("cmp eax, 0x800")
        self.assertEqual(instr.outputs, [Register("zf")])
        self.assertIn(Register("eax"), instr.inputs)

    def test_cmp_mem_immediate(self) -> None:
        instr = self.parse_instruction("cmp [ecx], 0x0")
        self.assertEqual(instr.outputs, [Register("zf")])
        self.assertIn(Register("ecx"), instr.inputs)

    def test_cmp_reg_memory(self) -> None:
        instr = self.parse_instruction("cmp edx, [ecx]")
        self.assertEqual(instr.outputs, [Register("zf")])
        self.assertIn(Register("edx"), instr.inputs)
        self.assertIn(Register("ecx"), instr.inputs)

    def test_cmp_reg_reg(self) -> None:
        instr = self.parse_instruction("cmp ebp, eax")
        self.assertEqual(instr.outputs, [Register("zf")])
        self.assertIn(Register("ebp"), instr.inputs)

    def test_jz_branch(self) -> None:
        instr = self.parse_instruction("jz _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("zf")])

    def test_jnz_branch(self) -> None:
        instr = self.parse_instruction("jnz _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("zf")])

    def test_ja_branch(self) -> None:
        instr = self.parse_instruction("ja _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("cf"), Register("zf")])

    def test_jl_branch(self) -> None:
        instr = self.parse_instruction("jl _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("sf"), Register("of")])

    def test_jle_branch(self) -> None:
        instr = self.parse_instruction("jle _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("zf"), Register("sf"), Register("of")])

    def test_jg_branch(self) -> None:
        instr = self.parse_instruction("jg _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("zf"), Register("sf"), Register("of")])

    def test_jge_branch(self) -> None:
        instr = self.parse_instruction("jge _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("sf"), Register("of")])

    def test_jns_branch(self) -> None:
        instr = self.parse_instruction("jns _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("sf")])

    def test_jc_branch(self) -> None:
        instr = self.parse_instruction("jc _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("cf")])

    def test_jnc_branch(self) -> None:
        instr = self.parse_instruction("jnc _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("cf")])

    def test_jbe_branch(self) -> None:
        instr = self.parse_instruction("jbe _target")
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")
        self.assertCountEqual(instr.inputs, [Register("cf"), Register("zf")])

    def test_jmp_absolute(self) -> None:
        instr = self.parse_instruction("jmp _target")
        self.assertTrue(instr.is_jump())
        self.assertFalse(instr.is_conditional)
        self.assertEqual(instr.jump_target.target, "_target")

    def test_jmp_register(self) -> None:
        instr = self.parse_instruction("jmp eax")
        self.assertTrue(instr.is_jump())
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target, Register("eax"))
        self.assertEqual(instr.inputs, [Register("eax")])

    def test_rep_movsd(self) -> None:
        instr = self.parse_instruction("rep movsd")
        self.assertTrue(instr.is_load)
        self.assertTrue(instr.is_store)
        self.assertCountEqual(instr.inputs, [Register("esi"), Register("edi"), Register("ecx")])
        self.assertCountEqual(instr.outputs, [Register("esi"), Register("edi"), Register("ecx")])

    def test_rep_stosd(self) -> None:
        instr = self.parse_instruction("rep stosd")
        self.assertFalse(instr.is_load)
        self.assertTrue(instr.is_store)
        self.assertCountEqual(instr.inputs, [Register("edi"), Register("eax"), Register("ecx")])
        self.assertCountEqual(instr.outputs, [Register("edi"), Register("ecx")])

    def test_repne_scasb(self) -> None:
        instr = self.parse_instruction("repne scasb")
        self.assertTrue(instr.is_load)
        self.assertFalse(instr.is_store)
        self.assertCountEqual(instr.inputs, [Register("edi"), Register("ecx"), Register("eax")])
        self.assertCountEqual(instr.outputs, [Register("edi"), Register("ecx"), Register("zf")])

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

    def test_lea_base_offset(self) -> None:
        instr = self.parse_instruction("lea eax, [esp + 0x4]")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("esp"), instr.inputs)

    def test_lea_scaled_index(self) -> None:
        instr = self.parse_instruction("lea eax, [edx*4 + 0x0]")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("edx"), instr.inputs)

    def test_mov_byte_load(self) -> None:
        instr, asm_state = self.parse_instruction_with_state("mov cl, byte ptr [eax]")
        self.assertIn(Register("eax"), instr.inputs)
        self.assertEqual(instr.outputs, [Register("ecx")])
        self.assertIn(
            "cl",
            asm_state.reg_formatter.aliases_for(instr.outputs[0]),
        )

    def test_mov_bl_load(self) -> None:
        instr, asm_state = self.parse_instruction_with_state("mov bl, byte ptr [esi + 0xbb]")
        self.assertIn(Register("esi"), instr.inputs)
        self.assertEqual(instr.outputs, [Register("ebx")])
        self.assertIn(
            "bl",
            asm_state.reg_formatter.aliases_for(instr.outputs[0]),
        )

    def test_mov_dx_word_stack_load(self) -> None:
        instr, asm_state = self.parse_instruction_with_state("mov dx, word ptr [esp + 0x8]")
        self.assertIn(Register("esp"), instr.inputs)
        self.assertEqual(instr.outputs, [Register("edx")])
        self.assertIn(
            "dx",
            asm_state.reg_formatter.aliases_for(instr.outputs[0]),
        )

    def test_mov_al_load(self) -> None:
        instr, asm_state = self.parse_instruction_with_state("mov al, byte ptr [esi]")
        self.assertIn(Register("esi"), instr.inputs)
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(
            "al",
            asm_state.reg_formatter.aliases_for(instr.outputs[0]),
        )

    def test_mov_word_load(self) -> None:
        instr, asm_state = self.parse_instruction_with_state("mov bp, word ptr [eax + 0x8]")
        self.assertIn(Register("eax"), instr.inputs)
        self.assertEqual(instr.outputs, [Register("ebp")])
        self.assertIn(
            "bp",
            asm_state.reg_formatter.aliases_for(instr.outputs[0]),
        )

    def test_mov_cl_immediate(self) -> None:
        instr, asm_state = self.parse_instruction_with_state("mov cl, 0xff")
        self.assertEqual(instr.outputs, [Register("ecx")])
        self.assertIn(
            "cl",
            asm_state.reg_formatter.aliases_for(instr.outputs[0]),
        )

    def test_mov_ch_immediate(self) -> None:
        instr, asm_state = self.parse_instruction_with_state("mov ch, 0xff")
        self.assertEqual(instr.outputs, [Register("ecx")])
        self.assertIn(
            "ch",
            asm_state.reg_formatter.aliases_for(instr.outputs[0]),
        )

    def test_and_register_immediate(self) -> None:
        instr = self.parse_instruction("and ah, 0xeb")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("eax"), instr.inputs)

    def test_or_register_immediate(self) -> None:
        instr = self.parse_instruction("or ecx, 0xffffffff")
        self.assertEqual(instr.outputs, [Register("ecx")])
        self.assertIn(Register("ecx"), instr.inputs)

    def test_dec_register(self) -> None:
        instr = self.parse_instruction("dec eax")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("eax"), instr.inputs)

    def test_inc_register(self) -> None:
        instr = self.parse_instruction("inc eax")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("eax"), instr.inputs)


if __name__ == "__main__":
    unittest.main()
