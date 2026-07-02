from __future__ import annotations

import unittest

from typing import Tuple

from m2c.arch_x86 import X86Arch
from m2c.asm_instruction import (
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    AsmState,
    BinOp,
    JumpTarget,
    RegFormatter,
    Register,
    parse_asm_instruction,
)
from m2c.instruction import Instruction, InstructionMeta, StackLocation


class TestX86Parsing(unittest.TestCase):
    """Parse-level tests for the x86 arch: mnemonic normalization (width
    suffixes, sub-register rewriting) and structural instruction information
    (inputs/outputs/is_load/is_store/jump_target)."""

    def setUp(self) -> None:
        self.arch = X86Arch()

    def parse_asm(self, line: str) -> Tuple[AsmInstruction, AsmState]:
        asm_state = AsmState(reg_formatter=RegFormatter())
        asm = parse_asm_instruction(line, self.arch, asm_state)
        return asm, asm_state

    def parse_instruction(self, line: str) -> Instruction:
        asm, _ = self.parse_asm(line)
        return self.arch.parse(asm.mnemonic, asm.args, InstructionMeta.missing())

    # Argument parsing

    def test_dword_ptr_address_mode(self) -> None:
        asm, _ = self.parse_asm("MOV EAX, dword ptr [ESP + 0xc]")
        # Size prefixes are folded into the mnemonic; dword is the default
        # width and adds no suffix.
        self.assertEqual(asm.mnemonic, "mov")
        self.assertEqual(len(asm.args), 2)
        addr = asm.args[1]
        assert isinstance(addr, AsmAddressMode)
        self.assertEqual(addr.base, Register("esp"))
        self.assertEqual(addr.addend, AsmLiteral(0xC))

    def test_byte_ptr_width_suffix(self) -> None:
        asm, _ = self.parse_asm("MOV byte ptr [EAX], CL")
        self.assertEqual(asm.mnemonic, "mov.b")
        addr = asm.args[0]
        assert isinstance(addr, AsmAddressMode)
        self.assertEqual(addr.base, Register("eax"))
        self.assertEqual(asm.args[1], Register("ecx"))

    def test_word_ptr_width_suffix(self) -> None:
        asm, _ = self.parse_asm("MOV DX, word ptr [ESP + 0x8]")
        self.assertEqual(asm.mnemonic, "mov.w")
        self.assertEqual(asm.args[0], Register("edx"))

    def test_qword_ptr_width_suffix(self) -> None:
        asm, _ = self.parse_asm("FSTP qword ptr [EBP + -0x10]")
        self.assertEqual(asm.mnemonic, "fstp.q")
        addr = asm.args[0]
        assert isinstance(addr, AsmAddressMode)
        self.assertEqual(addr.base, Register("ebp"))
        self.assertEqual(addr.addend, AsmLiteral(-0x10))

    def test_absolute_symbol_address_mode(self) -> None:
        asm, _ = self.parse_asm("MOV EAX, [_DAT_0079a8b0]")
        self.assertEqual(len(asm.args), 2)
        addr = asm.args[1]
        assert isinstance(addr, AsmAddressMode)
        self.assertIsNone(addr.base)
        self.assertEqual(addr.addend, AsmGlobalSymbol("_DAT_0079a8b0"))

    def test_absolute_literal_address_mode(self) -> None:
        asm, _ = self.parse_asm("CMP dword ptr [0x4a1b20], 0x0")
        addr = asm.args[0]
        assert isinstance(addr, AsmAddressMode)
        self.assertIsNone(addr.base)
        self.assertEqual(addr.addend, AsmLiteral(0x4A1B20))

    def test_scaled_index_address_mode(self) -> None:
        asm, _ = self.parse_asm("MOV dword ptr [ESI + EBX*0x8 + 0x30], ECX")
        addr = asm.args[0]
        assert isinstance(addr, AsmAddressMode)
        self.assertEqual(addr.base, Register("esi"))
        self.assertEqual(
            addr.addend,
            BinOp("+", BinOp("*", Register("ebx"), AsmLiteral(8)), AsmLiteral(0x30)),
        )

    def test_scaled_index_symbol_address_mode(self) -> None:
        asm, _ = self.parse_asm(
            "JMP dword ptr [EAX*0x4 + _switchD_0040100d_switchdataD_00401058]"
        )
        self.assertEqual(asm.mnemonic, "jmp")
        addr = asm.args[0]
        assert isinstance(addr, AsmAddressMode)
        self.assertIsNone(addr.base)
        self.assertEqual(
            addr.addend,
            BinOp(
                "+",
                BinOp("*", Register("eax"), AsmLiteral(4)),
                AsmGlobalSymbol("_switchD_0040100d_switchdataD_00401058"),
            ),
        )

    def test_offset_symbol(self) -> None:
        asm, _ = self.parse_asm("PUSH offset _FUN_0040e440")
        self.assertEqual(asm.mnemonic, "push")
        self.assertEqual(asm.args, [AsmGlobalSymbol("_FUN_0040e440")])

    def test_negative_displacement(self) -> None:
        asm, _ = self.parse_asm("MOV EAX, dword ptr [EBP + -0x8]")
        addr = asm.args[1]
        assert isinstance(addr, AsmAddressMode)
        self.assertEqual(addr.base, Register("ebp"))
        self.assertEqual(addr.addend, AsmLiteral(-8))

    def test_st_registers(self) -> None:
        asm, _ = self.parse_asm("FADDP ST(2), ST(0)")
        self.assertEqual(asm.args, [Register("st2"), Register("st0")])

    # Sub-register aliasing & width preservation

    def test_mov_byte_load_subregister(self) -> None:
        asm, _ = self.parse_asm("MOV CL, byte ptr [EAX]")
        self.assertEqual(asm.mnemonic, "mov.b")
        self.assertEqual(asm.args[0], Register("ecx"))

    def test_mov_bl_load(self) -> None:
        instr = self.parse_instruction("MOV BL, byte ptr [ESI + 0xbb]")
        self.assertEqual(instr.mnemonic, "mov.b")
        self.assertIn(Register("esi"), instr.inputs)
        self.assertEqual(instr.outputs, [Register("ebx")])
        self.assertTrue(instr.is_load)

    def test_mov_subregister_immediate(self) -> None:
        asm, _ = self.parse_asm("MOV CL, 0xff")
        self.assertEqual(asm.mnemonic, "mov.b")
        self.assertEqual(asm.args, [Register("ecx"), AsmLiteral(0xFF)])
        asm, _ = self.parse_asm("MOV CH, 0xff")
        self.assertEqual(asm.mnemonic, "mov.b")
        self.assertEqual(asm.args[0], Register("ecx"))

    def test_mov_word_subregister(self) -> None:
        asm, _ = self.parse_asm("MOV BP, word ptr [EAX + 0x8]")
        self.assertEqual(asm.mnemonic, "mov.w")
        self.assertEqual(asm.args[0], Register("ebp"))

    def test_movsx_subregister(self) -> None:
        asm, _ = self.parse_asm("MOVSX EDX, AX")
        # The width suffix reflects the narrowest (source) operand.
        self.assertEqual(asm.mnemonic, "movsx.w")
        self.assertEqual(asm.args, [Register("edx"), Register("eax")])

    def test_and_subregister(self) -> None:
        instr = self.parse_instruction("AND AH, 0xeb")
        self.assertEqual(instr.mnemonic, "and.b")
        self.assertEqual(instr.outputs[0], Register("eax"))
        self.assertIn(Register("eax"), instr.inputs)

    def test_subregister_names_preserved_in_formatter(self) -> None:
        _, asm_state = self.parse_asm("MOV CL, 0xff")
        # The formatter tracked the sub-register spelling that was used.
        self.assertEqual(asm_state.reg_formatter.format(Register("cl")), "cl")

    # Structural instruction information

    def test_mov_stack_load(self) -> None:
        instr = self.parse_instruction("MOV EAX, dword ptr [ESP + 0x4]")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertTrue(instr.is_load)
        self.assertFalse(instr.is_store)
        self.assertIn(Register("esp"), instr.inputs)
        self.assertIn(StackLocation(offset=4, symbolic_offset=None), instr.inputs)

    def test_mov_stack_store(self) -> None:
        instr = self.parse_instruction("MOV dword ptr [ESP + 0x8], EAX")
        self.assertTrue(instr.is_store)
        self.assertFalse(instr.is_load)
        self.assertIn(Register("eax"), instr.inputs)
        self.assertIn(StackLocation(offset=8, symbolic_offset=None), instr.outputs)

    def test_mov_base_offset_store(self) -> None:
        instr = self.parse_instruction("MOV dword ptr [EAX + 0x1c], ECX")
        self.assertTrue(instr.is_store)
        self.assertIn(Register("ecx"), instr.inputs)
        self.assertIn(Register("eax"), instr.inputs)
        self.assertEqual(instr.outputs, [])

    def test_mov_absolute_load(self) -> None:
        instr = self.parse_instruction("MOV ECX, [_DAT_00401000]")
        self.assertEqual(instr.outputs, [Register("ecx")])
        self.assertTrue(instr.is_load)
        self.assertEqual(instr.inputs, [])

    def test_mov_scaled_index_store(self) -> None:
        instr = self.parse_instruction("MOV dword ptr [ESI + EBX*0x8 + 0x30], ECX")
        self.assertTrue(instr.is_store)
        self.assertCountEqual(
            instr.inputs, [Register("ecx"), Register("esi"), Register("ebx")]
        )

    def test_push_register(self) -> None:
        instr = self.parse_instruction("PUSH EAX")
        self.assertEqual(instr.outputs, [Register("esp")])
        self.assertCountEqual(instr.inputs, [Register("esp"), Register("eax")])
        self.assertTrue(instr.is_store)

    def test_push_offset_symbol(self) -> None:
        instr = self.parse_instruction("PUSH offset _FUN_0040e440")
        self.assertEqual(instr.outputs, [Register("esp")])
        self.assertIn(Register("esp"), instr.inputs)

    def test_pop_register(self) -> None:
        instr = self.parse_instruction("POP ESI")
        self.assertEqual(instr.inputs, [Register("esp")])
        self.assertEqual(instr.outputs, [Register("esi"), Register("esp")])
        self.assertTrue(instr.is_load)

    def test_ret(self) -> None:
        instr = self.parse_instruction("RET")
        self.assertTrue(instr.is_return)
        self.assertTrue(instr.is_jump())
        self.assertEqual(instr.inputs, [Register("esp")])

    def test_ret_imm(self) -> None:
        instr = self.parse_instruction("RET 0x4")
        self.assertTrue(instr.is_return)

    def test_sub_esp_allocates_stack(self) -> None:
        instr = self.parse_instruction("SUB ESP, 0x8")
        self.assertIn(Register("esp"), instr.outputs)
        self.assertIn(Register("esp"), instr.inputs)

    def test_add_register_register(self) -> None:
        instr = self.parse_instruction("ADD EAX, EDX")
        self.assertIn(Register("eax"), instr.outputs)
        self.assertCountEqual(instr.inputs, [Register("eax"), Register("edx")])

    def test_add_memory_dst(self) -> None:
        instr = self.parse_instruction("ADD dword ptr [EAX + 0x8], ECX")
        self.assertTrue(instr.is_store)
        self.assertIn(Register("eax"), instr.inputs)
        self.assertIn(Register("ecx"), instr.inputs)

    def test_sbb_register_register(self) -> None:
        instr = self.parse_instruction("SBB EAX, EAX")
        self.assertIn(Register("eax"), instr.outputs)
        self.assertIn(Register("eax"), instr.inputs)
        # sbb reads the carry pseudo-flag
        for flag in self.arch.flag_regs:
            self.assertIn(flag, instr.outputs)

    def test_shifts(self) -> None:
        for mn in ("SHL", "SAL", "SHR", "SAR", "ROL", "ROR"):
            instr = self.parse_instruction(f"{mn} ECX, 0x8")
            self.assertIn(Register("ecx"), instr.outputs)
            self.assertIn(Register("ecx"), instr.inputs)

    def test_shrd(self) -> None:
        instr = self.parse_instruction("SHRD EAX, EDX, 0x8")
        self.assertIn(Register("eax"), instr.outputs)
        self.assertCountEqual(instr.inputs, [Register("edx"), Register("eax")])

    def test_unary_rmw(self) -> None:
        for mn in ("NEG", "NOT", "INC", "DEC", "BSWAP"):
            instr = self.parse_instruction(f"{mn} EAX")
            self.assertIn(Register("eax"), instr.outputs)
            self.assertEqual(instr.inputs, [Register("eax")])

    def test_cdq(self) -> None:
        instr = self.parse_instruction("CDQ")
        self.assertEqual(instr.outputs, [Register("edx")])
        self.assertEqual(instr.inputs, [Register("eax")])

    def test_mul_register(self) -> None:
        instr = self.parse_instruction("MUL ECX")
        self.assertIn(Register("eax"), instr.outputs)
        self.assertIn(Register("edx"), instr.outputs)
        self.assertCountEqual(instr.inputs, [Register("eax"), Register("ecx")])

    def test_imul_single_operand(self) -> None:
        instr = self.parse_instruction("IMUL ECX")
        self.assertIn(Register("eax"), instr.outputs)
        self.assertIn(Register("edx"), instr.outputs)
        self.assertCountEqual(instr.inputs, [Register("eax"), Register("ecx")])

    def test_imul_memory_single(self) -> None:
        instr = self.parse_instruction("IMUL dword ptr [EBP + -0xc]")
        self.assertIn(Register("eax"), instr.outputs)
        self.assertIn(Register("edx"), instr.outputs)
        self.assertIn(Register("eax"), instr.inputs)
        self.assertIn(Register("ebp"), instr.inputs)
        self.assertTrue(instr.is_load)

    def test_imul_reg_memory(self) -> None:
        instr = self.parse_instruction("IMUL EDI, dword ptr [ESP + 0x54]")
        self.assertIn(Register("edi"), instr.outputs)
        self.assertIn(Register("edi"), instr.inputs)
        self.assertIn(Register("esp"), instr.inputs)
        self.assertTrue(instr.is_load)

    def test_imul_reg_reg(self) -> None:
        instr = self.parse_instruction("IMUL EDI, EBX")
        self.assertIn(Register("edi"), instr.outputs)
        self.assertCountEqual(instr.inputs, [Register("edi"), Register("ebx")])

    def test_idiv_register(self) -> None:
        instr = self.parse_instruction("IDIV ECX")
        self.assertIn(Register("eax"), instr.outputs)
        self.assertIn(Register("edx"), instr.outputs)
        self.assertCountEqual(
            instr.inputs, [Register("eax"), Register("edx"), Register("ecx")]
        )

    def test_idiv_absolute_memory(self) -> None:
        instr = self.parse_instruction("IDIV dword ptr [_DAT_00667c1c]")
        self.assertIn(Register("eax"), instr.inputs)
        self.assertIn(Register("edx"), instr.inputs)
        self.assertTrue(instr.is_load)

    def test_xor_zeroes_register(self) -> None:
        instr = self.parse_instruction("XOR EAX, EAX")
        self.assertIn(Register("eax"), instr.outputs)
        self.assertIn(Register("eax"), instr.inputs)

    def test_test_sets_flags(self) -> None:
        instr = self.parse_instruction("TEST EAX, EAX")
        self.assertEqual(instr.inputs, [Register("eax")])
        self.assertCountEqual(instr.outputs, self.arch.flag_regs)

    def test_test_stack_memory_immediate(self) -> None:
        instr = self.parse_instruction("TEST dword ptr [ESP + 0x8], 0x2")
        self.assertIn(Register("esp"), instr.inputs)
        self.assertCountEqual(instr.outputs, self.arch.flag_regs)

    def test_cmp_variants(self) -> None:
        for line in (
            "CMP EAX, 0x1b5",
            "CMP dword ptr [ECX], 0x0",
            "CMP dword ptr [_DAT_00668fb8], 0x0",
            "CMP EAX, offset _dat_00829ae4",
            "CMP EDX, dword ptr [ECX]",
            "CMP EBP, EAX",
        ):
            instr = self.parse_instruction(line)
            self.assertCountEqual(instr.outputs, self.arch.flag_regs)
            self.assertFalse(instr.is_store)

    def test_lea_base_offset(self) -> None:
        instr = self.parse_instruction("LEA EAX, [ESP + 0x4]")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("esp"), instr.inputs)
        # lea computes an address without loading through it
        self.assertFalse(instr.is_load)

    def test_lea_scaled_index(self) -> None:
        instr = self.parse_instruction("LEA EAX, [EDX*0x4 + 0x0]")
        self.assertEqual(instr.outputs, [Register("eax")])
        self.assertIn(Register("edx"), instr.inputs)

    # Calls

    def test_call_symbol(self) -> None:
        instr = self.parse_instruction("CALL _malloc")
        self.assertEqual(instr.function_target, AsmGlobalSymbol("_malloc"))
        self.assertEqual(instr.outputs, [Register("eax"), Register("edx")])

    def test_call_import_symbol(self) -> None:
        instr = self.parse_instruction("CALL __imp__MultiByteToWideChar_24")
        self.assertEqual(
            instr.function_target, AsmGlobalSymbol("__imp__MultiByteToWideChar_24")
        )

    def test_call_memory(self) -> None:
        instr = self.parse_instruction("CALL dword ptr [ECX + 0x8]")
        self.assertEqual(instr.inputs, [Register("ecx")])
        self.assertTrue(instr.is_load)
        self.assertEqual(instr.outputs, [Register("eax"), Register("edx")])

    # Jumps

    def test_jmp_label(self) -> None:
        instr = self.parse_instruction("JMP _target")
        self.assertTrue(instr.is_jump())
        self.assertFalse(instr.is_conditional)
        self.assertEqual(instr.jump_target, JumpTarget("_target"))

    def test_jmp_register(self) -> None:
        instr = self.parse_instruction("JMP EAX")
        self.assertTrue(instr.is_jump())
        self.assertTrue(instr.is_conditional)
        self.assertEqual(instr.jump_target, Register("eax"))
        self.assertEqual(instr.inputs, [Register("eax")])

    def test_jmp_jump_table(self) -> None:
        instr = self.parse_instruction(
            "JMP dword ptr [EAX*0x4 + _switchD_0040100d_switchdataD_00401058]"
        )
        self.assertTrue(instr.is_jump())
        self.assertTrue(instr.is_conditional)
        self.assertTrue(instr.is_load)
        self.assertEqual(instr.jump_target, Register("eax"))
        self.assertEqual(instr.inputs, [Register("eax")])

    def test_conditional_jumps(self) -> None:
        # x86 condition codes map onto ARM-style flag pseudo-registers.
        cases = {
            "JZ": "z",
            "JNZ": "z",
            "JS": "n",
            "JNS": "n",
            "JC": "c",
            "JNC": "c",
            "JA": "hi",
            "JBE": "hi",
            "JL": "ge",
            "JGE": "ge",
            "JG": "gt",
            "JLE": "gt",
        }
        for mn, flag in cases.items():
            instr = self.parse_instruction(f"{mn} _target")
            self.assertTrue(instr.is_conditional, mn)
            self.assertEqual(instr.jump_target, JumpTarget("_target"), mn)
            self.assertEqual(instr.inputs, [Register(flag)], mn)

    def test_setcc(self) -> None:
        cases = {
            "SETZ": "z",
            "SETNZ": "z",
            "SETG": "gt",
            "SETGE": "ge",
            "SETL": "ge",
            "SETBE": "hi",
        }
        for mn, flag in cases.items():
            instr = self.parse_instruction(f"{mn} AL")
            self.assertEqual(instr.inputs, [Register(flag)], mn)
            self.assertEqual(instr.outputs, [Register("eax")], mn)

    # String instructions

    def test_rep_movsd(self) -> None:
        instr = self.parse_instruction("REP MOVSD")
        self.assertEqual(instr.mnemonic, "rep.movsd")
        self.assertTrue(instr.is_load)
        self.assertTrue(instr.is_store)
        regs = [Register("esi"), Register("edi"), Register("ecx")]
        self.assertCountEqual(instr.inputs, regs)
        self.assertCountEqual(instr.outputs, regs)

    def test_rep_stosd(self) -> None:
        instr = self.parse_instruction("REP STOSD")
        self.assertFalse(instr.is_load)
        self.assertTrue(instr.is_store)
        self.assertCountEqual(
            instr.inputs, [Register("edi"), Register("eax"), Register("ecx")]
        )
        self.assertCountEqual(instr.outputs, [Register("edi"), Register("ecx")])

    def test_repne_scasb(self) -> None:
        instr = self.parse_instruction("REPNE SCASB")
        self.assertTrue(instr.is_load)
        self.assertFalse(instr.is_store)
        self.assertCountEqual(
            instr.inputs, [Register("edi"), Register("ecx"), Register("eax")]
        )
        self.assertCountEqual(
            instr.outputs, [Register("edi"), Register("ecx"), Register("z")]
        )

    # Unknown instructions parse without crashing (x87 FPU etc.)

    def test_unknown_instructions_parse(self) -> None:
        for line in (
            "FLD dword ptr [ESP + 0x10]",
            "FSTP dword ptr [_DAT_004b5c5c]",
            "FADDP ST(1), ST(0)",
            "FNSTSW AX",
            "RDTSC",
            "FILD dword ptr [ESP]",
        ):
            instr = self.parse_instruction(line)
            self.assertIsInstance(instr, Instruction)


class TestX86AsmFile(unittest.TestCase):
    """File-level parsing: labels, functions, and .long jump tables."""

    ASM = """
_DRIVING_SCHOOL_FUN_00401000:
    MOV EAX, dword ptr [ESP + 0xc]
    CMP EAX, 0x6
    JA _switchD_0040100d_caseD_2
_switchD_0040100d_switchD:
    JMP dword ptr [EAX*0x4 + _switchD_0040100d_switchdataD_00401058]
_switchD_0040100d_caseD_1:
    MOV EDX, dword ptr [ESP + 0xc]
    RET
_switchD_0040100d_caseD_2:
    MOV EDX, dword ptr [ESP + 0x4]
    RET
_switchD_0040100d_switchdataD_00401058:
    .long _switchD_0040100d_caseD_1, _switchD_0040100d_caseD_2
"""

    def test_parse_file_with_long_jump_table(self) -> None:
        import io

        from m2c.asm_file import AsmSymbolicData, parse_file
        from m2c.main import parse_flags

        options = parse_flags(["--target", "x86-gcc-c", "irrelevant.s"])
        f = io.StringIO(self.ASM)
        f.name = "test.s"  # parse_file reads f.name
        asm_file = parse_file(f, X86Arch(), options)

        self.assertEqual(len(asm_file.functions), 1)
        self.assertEqual(asm_file.functions[0].name, "_DRIVING_SCHOOL_FUN_00401000")

        # The .long directive produced 4-byte symbolic jump table entries.
        entry = asm_file.asm_data.values["_switchD_0040100d_switchdataD_00401058"]
        symbols = [
            d.as_symbol_without_addend() if isinstance(d, AsmSymbolicData) else None
            for d in entry.data
        ]
        self.assertEqual(
            symbols,
            ["_switchD_0040100d_caseD_1", "_switchD_0040100d_caseD_2"],
        )


if __name__ == "__main__":
    unittest.main()
