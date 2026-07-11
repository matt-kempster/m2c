from __future__ import annotations

import unittest

from typing import Dict, List, Optional, Set, Tuple

from m2c.asm_pattern import BodyPart

from m2c.arch_x86 import X86Arch
from m2c.error import DecompFailure
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
from m2c.types import Type


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

    def test_branch_distance_hints(self) -> None:
        # IDA/MASM-style branch-distance keywords are pure syntax.
        asm, _ = self.parse_asm("jl short loc_685729")
        self.assertEqual(asm.mnemonic, "jl")
        self.assertEqual(asm.args, [AsmGlobalSymbol("loc_685729")])
        asm, _ = self.parse_asm("call near ptr _foo")
        self.assertEqual(asm.mnemonic, "call")
        self.assertEqual(asm.args, [AsmGlobalSymbol("_foo")])

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
        self.assertEqual(asm.args[0], Register("ch"))

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
        self.assertEqual(instr.outputs[0], Register("ah"))
        self.assertIn(Register("ah"), instr.inputs)

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

    def test_base_less_memory_not_stack_access(self) -> None:
        # A base-less Intel memory operand -- an absolute [symbol] or a
        # scaled-index expression with no plain base register (base=None) --
        # must not be modeled as a stack access: no StackLocation should appear
        # among its inputs/outputs (only esp-relative operands are stack).
        load = self.parse_instruction("MOV EAX, [_DAT_00401000]")
        self.assertFalse(
            any(isinstance(loc, StackLocation) for loc in load.inputs + load.outputs)
        )
        store = self.parse_instruction("MOV dword ptr [ESI*0x4 + _table], EAX")
        self.assertTrue(store.is_store)
        self.assertIn(Register("esi"), store.inputs)
        self.assertFalse(
            any(isinstance(loc, StackLocation) for loc in store.inputs + store.outputs)
        )
        # Contrast: a genuine esp-relative store does produce a stack location.
        stack = self.parse_instruction("MOV dword ptr [ESP + 0x8], EAX")
        self.assertTrue(any(isinstance(loc, StackLocation) for loc in stack.outputs))

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
        self.assertIn(Register("c"), instr.inputs)
        for flag in self.arch.flag_regs:
            self.assertIn(flag, instr.outputs)

    def test_adc_reads_carry(self) -> None:
        instr = self.parse_instruction("ADC EAX, ECX")
        self.assertIn(Register("c"), instr.inputs)
        for flag in self.arch.flag_regs:
            self.assertIn(flag, instr.outputs)

    def test_inc_dec_preserve_carry(self) -> None:
        # inc/dec set all flags except the carry flag (real x86 semantics).
        for mn in ("INC", "DEC"):
            instr = self.parse_instruction(f"{mn} EAX")
            self.assertNotIn(Register("c"), instr.outputs, mn)
            self.assertNotIn(Register("c"), instr.clobbers, mn)
            self.assertIn(Register("z"), instr.outputs, mn)
            self.assertIn(Register("gt"), instr.outputs, mn)

    def test_rotates_clobber_flags(self) -> None:
        for mn in ("ROL", "ROR"):
            instr = self.parse_instruction(f"{mn} EAX, 0x4")
            self.assertNotIn(Register("z"), instr.outputs, mn)
            self.assertIn(Register("z"), instr.clobbers, mn)

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
        for mn in ("NEG", "NOT", "BSWAP"):
            instr = self.parse_instruction(f"{mn} EAX")
            self.assertIn(Register("eax"), instr.outputs)
            self.assertEqual(instr.inputs, [Register("eax")])
        # inc/dec additionally read the carry flag: they preserve it, but fold
        # it into the composite unsigned-above (ja/jbe) predicate.
        for mn in ("INC", "DEC"):
            instr = self.parse_instruction(f"{mn} EAX")
            self.assertIn(Register("eax"), instr.outputs)
            self.assertCountEqual(instr.inputs, [Register("eax"), Register("c")])

    def test_cdq(self) -> None:
        instr = self.parse_instruction("CDQ")
        self.assertEqual(instr.outputs, [Register("edx")])
        self.assertEqual(instr.inputs, [Register("eax")])

    def test_mul_register(self) -> None:
        instr = self.parse_instruction("MUL ECX")
        self.assertIn(Register("eax"), instr.outputs)
        self.assertIn(Register("edx"), instr.outputs)
        self.assertCountEqual(instr.inputs, [Register("eax"), Register("ecx")])
        # mul/div leave the flags in an unusable state; they are clobbered
        # rather than given symbolic values.
        self.assertCountEqual(instr.clobbers, self.arch.flag_regs)

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
_switch_func_00401000:
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
        self.assertEqual(asm_file.functions[0].name, "_switch_func_00401000")

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


class TestX86FunctionAbi(unittest.TestCase):
    """The i386 cdecl/stdcall stack argument layout. Every stack argument
    occupies 4-byte slots regardless of natural alignment, so an 8-byte
    `double` starts at a 4-byte boundary (not an 8-byte one)."""

    def setUp(self) -> None:
        self.arch = X86Arch()

    def abi_offsets(
        self, params: List[Tuple[Type, str]]
    ) -> List[Tuple[Optional[int], Optional[str]]]:
        from m2c.types import FunctionParam, FunctionSignature

        sig = FunctionSignature(
            return_type=Type.void(),
            params=[FunctionParam(type=t, name=n) for t, n in params],
            params_known=True,
            is_variadic=False,
        )
        abi = self.arch.function_abi(sig, {}, for_call=True)
        return [(slot.loc.offset, slot.name) for slot in abi.arg_slots]

    def test_double_returns_double(self) -> None:
        from m2c.types import Type

        # A leading `double` argument is passed at [esp+4], not [esp+8].
        self.assertEqual(self.abi_offsets([(Type.f64(), "d")]), [(4, "d")])

    def test_double_then_int(self) -> None:
        from m2c.types import Type

        # The double occupies [esp+4]..[esp+8]; the int follows at [esp+12].
        self.assertEqual(
            self.abi_offsets([(Type.f64(), "d"), (Type.s32(), "i")]),
            [(4, "d"), (12, "i")],
        )

    def test_int_then_double(self) -> None:
        from m2c.types import Type

        # The int is at [esp+4]; the double is 4-byte aligned at [esp+8]
        # (it is NOT bumped to [esp+12] for 8-byte alignment).
        self.assertEqual(
            self.abi_offsets([(Type.s32(), "i"), (Type.f64(), "d")]),
            [(4, "i"), (8, "d")],
        )

    def test_stdcall_double_cleanup_bytes(self) -> None:
        # A stdcall variant whose decorated cleanup bytes include an 8-byte
        # argument: `void f(int, double)` decorates as `_f@12` (4 for the int
        # + 8 for the double), which the import-thunk name `__imp__f_12`
        # carries. The callee pops 12 bytes, and its arguments still land in
        # 4-byte slots (i@4, d@8).
        from m2c.arch_x86 import callee_cleanup_bytes
        from m2c.asm_instruction import AsmGlobalSymbol

        self.assertEqual(callee_cleanup_bytes(AsmGlobalSymbol("__imp__f_12"), {}), 12)
        self.assertEqual(
            self.abi_offsets([(Type.s32(), "i"), (Type.f64(), "d")]),
            [(4, "i"), (8, "d")],
        )


class TestX86CalleeCleanupPrecedence(unittest.TestCase):
    """callee_cleanup_bytes resolves a call's stdcall cleanup with the
    precedence context > decorated @N > file `.set` decoration; structural
    inference (compute_call_cleanup) runs only when all three pass."""

    def cleanup(
        self,
        sym: str,
        context: Optional[Dict[str, int]] = None,
        file: Optional[Dict[str, int]] = None,
    ) -> Optional[int]:
        from m2c.arch_x86 import callee_cleanup_bytes

        return callee_cleanup_bytes(AsmGlobalSymbol(sym), context or {}, file or {})

    def test_undecorated_name_unknown(self) -> None:
        # An undecorated name with no context or file decoration resolves to
        # None: the convention is left to structural inference.
        self.assertIsNone(self.cleanup("_MessageBoxA"))

    def test_context_for_undecorated_name(self) -> None:
        # A context declaration supplies the cleanup for an undecorated name.
        self.assertEqual(self.cleanup("_MessageBoxA", context={"_MessageBoxA": 16}), 16)

    def test_context_overrides_decorated(self) -> None:
        # Context also outranks an `__imp__X_N` decorated suffix.
        self.assertEqual(self.cleanup("__imp__Foo_8"), 8)  # decorated
        self.assertEqual(self.cleanup("__imp__Foo_8", context={"__imp__Foo_8": 12}), 12)

    def test_decorated_overrides_file(self) -> None:
        # A decorated suffix outranks a file `.set` decoration.
        self.assertEqual(self.cleanup("__imp__Foo_8", file={"__imp__Foo_8": 99}), 8)

    def test_file_decoration_below_context(self) -> None:
        # A `.set name@N` file decoration is used, but context wins.
        self.assertEqual(self.cleanup("_Bar", file={"_Bar": 8}), 8)
        self.assertEqual(self.cleanup("_Bar", context={"_Bar": 4}, file={"_Bar": 8}), 4)


class TestX86StackRewrite(unittest.TestCase):
    """The ESP-delta stack-rewrite prepass, focused on stdcall/indirect
    cleanup inference not miscounting prologue/callee-save pushes as call
    arguments."""

    def setUp(self) -> None:
        self.arch = X86Arch()

    def _build_body(self, lines: str) -> Tuple[List[BodyPart], Set[str]]:
        from m2c.asm_file import Label

        asm_state = AsmState(reg_formatter=RegFormatter())
        body: List[BodyPart] = []
        labels: Set[str] = set()
        for raw in lines.strip().splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.endswith(":"):
                body.append(Label([line[:-1]]))
                labels.add(line[:-1])
                continue
            asm = parse_asm_instruction(line, self.arch, asm_state)
            body.append(
                self.arch.parse(asm.mnemonic, asm.args, InstructionMeta.missing())
            )
        return body, labels

    def rewrite(self, lines: str) -> List[Instruction]:
        from m2c.arch_x86 import rewrite_stack_ops
        from m2c.asm_file import AsmData

        body, labels = self._build_body(lines)
        out = rewrite_stack_ops(body, self.arch, AsmData(), labels)
        return [p for p in out if isinstance(p, Instruction)]

    def call_cleanup_bytes(self, instrs: List[Instruction]) -> int:
        """The inferred callee-cleanup byte count on the (single) rewritten
        call: the third literal argument, negative when cdecl/unknown."""
        calls = [p for p in instrs if p.mnemonic == "call"]
        self.assertEqual(len(calls), 1)
        consume = calls[0].args[2]
        assert isinstance(consume, AsmLiteral)
        return consume.value

    def test_indirect_call_after_prologue(self) -> None:
        # A `call eax` right after `push ebp; mov ebp, esp; push esi` must not
        # count the frame-pointer save or the callee-save push as arguments.
        instrs = self.rewrite(
            """
            PUSH EBP
            MOV EBP, ESP
            PUSH ESI
            CALL EAX
            POP ESI
            LEAVE
            RET
            """
        )
        # Cdecl/unknown (negative), NOT 8 (the two save pushes).
        self.assertTrue(self.call_cleanup_bytes(instrs) < 0)

    def test_indirect_call_after_callee_save(self) -> None:
        # Callee-save pushes (esi/edi) before an indirect call, restored in the
        # epilogue, are saves and not outgoing arguments.
        instrs = self.rewrite(
            """
            PUSH ESI
            PUSH EDI
            CALL EAX
            POP EDI
            POP ESI
            RET
            """
        )
        self.assertTrue(self.call_cleanup_bytes(instrs) < 0)

    def test_indirect_call_no_stack_args(self) -> None:
        # A register-indirect call with no outgoing arguments at all.
        instrs = self.rewrite(
            """
            PUSH EBP
            MOV EBP, ESP
            SUB ESP, 0x8
            CALL EAX
            LEAVE
            RET
            """
        )
        self.assertTrue(self.call_cleanup_bytes(instrs) < 0)

    def test_indirect_call_real_args_still_counted(self) -> None:
        # Control: genuine argument pushes (scratch registers, not saves) are
        # still counted, so a stdcall-like indirect callee's cleanup is inferred.
        instrs = self.rewrite(
            """
            PUSH EAX
            PUSH ECX
            CALL EDX
            RET
            """
        )
        self.assertEqual(self.call_cleanup_bytes(instrs), 8)

    def test_stdcall_dll_import_pops_arguments(self) -> None:
        # A genuine Win32-style stdcall DLL import (`_LIB_DLL_Func`) with no
        # caller cleanup after the call pops its own arguments.
        instrs = self.rewrite(
            """
            PUSH EAX
            CALL _USER32_DLL_Foo
            RET
            """
        )
        self.assertEqual(self.call_cleanup_bytes(instrs), 4)

    def test_cdecl_dll_export_not_double_counted(self) -> None:
        # A cdecl export that merely lives in a system DLL (e.g. the variadic
        # `_MSVCRT_DLL_sprintf`) matches the DLL-import name pattern but is
        # cleaned up by the caller (`add esp, N` after the call). Counting it as
        # a callee pop *and* honoring the caller's `add esp, 4` over-pops esp,
        # unbalancing the frame at the return. With the caller-cleanup guard the
        # call is treated as cdecl and the body balances (no DecompFailure).
        instrs = self.rewrite(
            """
            PUSH EAX
            CALL _MSVCRT_DLL_sprintf
            ADD ESP, 0x4
            RET
            """
        )
        self.assertTrue(any(p.mnemonic == "ret" for p in instrs))
        # The caller cleans up its 4 argument bytes; the callee pops nothing.
        self.assertEqual(self.call_cleanup_bytes(instrs), 4)


class TestX86FpuRewrite(unittest.TestCase):
    """The x87 stack-elimination prepass: depth dataflow and the rewrite of
    stack-relative st(i) names into flat virtual registers f0..f7."""

    def setUp(self) -> None:
        self.arch = X86Arch()

    def _build_body(self, lines: str) -> Tuple[List[BodyPart], Set[str]]:
        from m2c.asm_file import Label

        asm_state = AsmState(reg_formatter=RegFormatter())
        body: List[BodyPart] = []
        labels: Set[str] = set()
        for raw in lines.strip().splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.endswith(":"):
                body.append(Label([line[:-1]]))
                labels.add(line[:-1])
                continue
            asm = parse_asm_instruction(line, self.arch, asm_state)
            body.append(
                self.arch.parse(asm.mnemonic, asm.args, InstructionMeta.missing())
            )
        return body, labels

    def rewrite(
        self, lines: str, call_deltas: Optional[Dict[str, int]] = None
    ) -> List[str]:
        """Parse a small body (labels end with ':') and run the FPU prepass,
        returning the rewritten Instructions as strings."""
        from m2c.asm_file import AsmData
        from m2c.x86_fpu import rewrite_fpu_ops

        body, labels = self._build_body(lines)
        out = rewrite_fpu_ops(body, self.arch, AsmData(), labels, call_deltas or {})
        return [str(p) for p in out if isinstance(p, Instruction)]

    def infer(self, lines: str) -> List[str]:
        """Run the whole FPU pattern (with structural call-delta inference)."""
        from m2c.asm_file import AsmData
        from m2c.asm_pattern import AsmMatcher
        from m2c.x86_fpu import X86FpuRewritePattern

        body, labels = self._build_body(lines)
        matcher = AsmMatcher(self.arch, AsmData(), body, labels)
        repl = X86FpuRewritePattern().match(matcher)
        assert repl is not None
        return [str(p) for p in repl.new_body if isinstance(p, Instruction)]

    def test_straight_line_depth(self) -> None:
        # A push defines f{depth}; arithmetic keeps the top's name.
        out = self.rewrite(
            """
            FLD dword ptr [ESP + 0x4]
            FADD dword ptr [ESP + 0x8]
            FLD dword ptr [ESP + 0xc]
            FMULP ST(1), ST(0)
            FSTP dword ptr [_g]
            RET
            """
        )
        self.assertEqual(out[0], "fld $f0, 0x4($esp)")
        self.assertEqual(out[1], "fadd $f0, 0x8($esp)")
        self.assertEqual(out[2], "fld $f1, 0xc($esp)")
        # fmulp st(1),st(0): f0 = f0 * f1, pop f1.
        self.assertEqual(out[3], "fmulp $f0, $f1")
        self.assertEqual(out[4], "fstp [_g], $f0")

    def test_discard_pop(self) -> None:
        # `fstp st(0)` is a pure pop-discard.
        out = self.rewrite(
            """
            FLD dword ptr [ESP + 0x4]
            FLD dword ptr [ESP + 0x8]
            FSTP ST(0)
            FSTP dword ptr [_g]
            RET
            """
        )
        self.assertEqual(out[2], "fpop $f1")
        self.assertEqual(out[3], "fstp [_g], $f0")

    def test_fxch_swaps_slots(self) -> None:
        out = self.rewrite(
            """
            FLD dword ptr [ESP + 0x4]
            FLD dword ptr [ESP + 0x8]
            FXCH
            FSTP dword ptr [_g]
            RET
            """
        )
        # Bare fxch swaps st0 (f1) and st1 (f0); depth is unchanged.
        self.assertEqual(out[2], "fxch $f1, $f0")
        self.assertEqual(out[3], "fstp [_g], $f1")

    def test_fld_st_duplicates_top(self) -> None:
        out = self.rewrite(
            """
            FLD dword ptr [ESP + 0x4]
            FLD ST(0)
            FADD dword ptr [ESP + 0x8]
            FMULP ST(1), ST(0)
            FSTP dword ptr [_g]
            RET
            """
        )
        # fld st(0) duplicates f0 into the new top f1.
        self.assertEqual(out[1], "fmov $f1, $f0")

    def test_merge_equal_depths(self) -> None:
        # Both branches reach L at depth 1: consistent.
        out = self.rewrite(
            """
            MOV EAX, dword ptr [ESP + 0x4]
            FLD dword ptr [ESP + 0x8]
            TEST EAX, EAX
            JZ L
            FADD dword ptr [ESP + 0xc]
            L:
            FSTP dword ptr [_g]
            RET
            """
        )
        self.assertIn("fstp [_g], $f0", out)

    def test_merge_conflict_raises(self) -> None:
        # One path pushes once, the other not at all: depth mismatch at the
        # label (each path's return depth stays <= 1, so this exercises the
        # merge check rather than the return-depth check).
        with self.assertRaises(DecompFailure) as cm:
            self.rewrite(
                """
                MOV EAX, dword ptr [ESP + 0x4]
                TEST EAX, EAX
                JZ L
                FLD dword ptr [ESP + 0x8]
                L:
                RET
                """
            )
        self.assertIn("depth mismatch", str(cm.exception))

    def test_return_depth_raises(self) -> None:
        # Two live values remain on the x87 stack at the return: the ABI allows
        # at most one (a float/double result in st(0)), so this fails loud.
        with self.assertRaises(DecompFailure) as cm:
            self.rewrite(
                """
                FLD dword ptr [ESP + 0x4]
                FLD dword ptr [ESP + 0x8]
                RET
                """
            )
        self.assertIn("not empty at return", str(cm.exception))

    def test_loop_preserves_depth(self) -> None:
        # A back-edge that keeps the stack balanced converges.
        out = self.rewrite(
            """
            FLDZ
            L:
            FADD dword ptr [ESP + 0x4]
            MOV EAX, dword ptr [ESP + 0x8]
            TEST EAX, EAX
            JNZ L
            FSTP dword ptr [_g]
            RET
            """
        )
        self.assertEqual(out[0], "fldz $f0")
        self.assertIn("fadd $f0, 0x4($esp)", out)

    def test_compare_pops(self) -> None:
        # fcompp pops both compared slots (depth 2 -> 0); the following push
        # therefore lands back in f0.
        out = self.rewrite(
            """
            FLD dword ptr [ESP + 0x4]
            FLD dword ptr [ESP + 0x8]
            FCOMPP
            FLD dword ptr [ESP + 0xc]
            FSTP dword ptr [_g]
            RET
            """
        )
        # fcompp compares st0 (f1) with st1 (f0).
        self.assertEqual(out[2], "fcompp $f1, $f0")
        # The next fld reuses f0 (stack was fully popped).
        self.assertEqual(out[3], "fld $f0, 0xc($esp)")

    def test_underflow_raises(self) -> None:
        with self.assertRaises(DecompFailure) as cm:
            self.rewrite(
                """
                FSTP dword ptr [_g]
                RET
                """
            )
        self.assertIn("underflow", str(cm.exception))

    def test_call_delta_plus_one_annotation(self) -> None:
        # A float-returning callee (delta +1) pushes a new top; the call is
        # annotated with the pushed virtual register (here f0) and the
        # following fadd operates at depth 1.
        out = self.rewrite(
            """
            CALL _foo, 0x0, 0x0
            FADD dword ptr [ESP + 0x4]
            FSTP dword ptr [_g]
            RET
            """,
            call_deltas={"_foo": 1},
        )
        self.assertEqual(out[0], "call _foo, 0x0, 0x0, 0x0, -0x1")
        self.assertEqual(out[1], "fadd $f0, 0x4($esp)")

    def test_ci_pow_two_arg_helper(self) -> None:
        # `fld a; fld b; call __CIpow` consumes st0 and st1 and pushes one result
        # (net -1). It rewrites to the fictive two-operand op reading st0 (f1)
        # and st1 (f0) -- not the ±1 single-consume annotation that would drop an
        # operand -- so the following store sees the pow result in f0.
        out = self.rewrite(
            """
            FLD qword ptr [ESP + 0x4]
            FLD qword ptr [ESP + 0xc]
            CALL __CIpow, 0x0, 0x0
            FSTP qword ptr [_g]
            RET
            """
        )
        self.assertEqual(out[2], "ci_pow.fictive $f1, $f0")
        self.assertEqual(out[3], "fstp.q [_g], $f0")

    def test_ci_sqrt_one_arg_helper(self) -> None:
        # `fld a; call __CIsqrt` consumes st0 and pushes one result in place
        # (net 0), rewritten to the fictive one-operand op.
        out = self.rewrite(
            """
            FLD qword ptr [ESP + 0x4]
            CALL __CIsqrt, 0x0, 0x0
            FSTP qword ptr [_g]
            RET
            """
        )
        self.assertEqual(out[1], "ci_sqrt.fictive $f0")

    def test_ci_pow_tail_call(self) -> None:
        # `fld a; fld b; jmp __CIpow` (the stack pass turns the jmp into a
        # `tailcall.fictive`): `return pow(a, b);`. The prepass emits the same
        # fictive op as a normal CI call, followed by a plain return, so the
        # pow result in f0 becomes the return value.
        from m2c.asm_file import AsmData
        from m2c.asm_instruction import AsmGlobalSymbol
        from m2c.x86_fpu import rewrite_fpu_ops

        body, labels = self._build_body(
            """
            FLD qword ptr [ESP + 0x4]
            FLD qword ptr [ESP + 0xc]
            """
        )
        body.append(
            self.arch.parse(
                "tailcall.fictive",
                [AsmGlobalSymbol("__CIpow")],
                InstructionMeta.missing(),
            )
        )
        out = [
            str(p)
            for p in rewrite_fpu_ops(body, self.arch, AsmData(), labels, {})
            if isinstance(p, Instruction)
        ]
        self.assertEqual(out[2], "ci_pow.fictive $f1, $f0")
        self.assertEqual(out[3], "ret")

    def test_ci_tail_call_wrong_depth_fails_loud(self) -> None:
        # A tail call to a two-argument helper with only one value on the x87
        # stack is unbalanced; fail loud rather than drop/invent an operand.
        from m2c.asm_file import AsmData
        from m2c.asm_instruction import AsmGlobalSymbol
        from m2c.x86_fpu import rewrite_fpu_ops

        body, labels = self._build_body("FLD qword ptr [ESP + 0x4]")
        body.append(
            self.arch.parse(
                "tailcall.fictive",
                [AsmGlobalSymbol("__CIpow")],
                InstructionMeta.missing(),
            )
        )
        with self.assertRaises(DecompFailure) as cm:
            rewrite_fpu_ops(body, self.arch, AsmData(), labels, {})
        self.assertIn("CRT math helper", str(cm.exception))

    def test_ci_unknown_helper_fails_loud(self) -> None:
        # An unrecognized `__CI*` helper must fail loud rather than let the ±1
        # inference guess a depth-consistent but value-wrong stack effect.
        with self.assertRaises(DecompFailure) as cm:
            self.rewrite(
                """
                FLD qword ptr [ESP + 0x4]
                CALL __CIbogus, 0x0, 0x0
                RET
                """
            )
        self.assertIn("__CIbogus", str(cm.exception))

    def test_context_float_return_wrapper_no_fpu(self) -> None:
        # A forwarding wrapper `call _returns_float; ret` has no x87 instruction
        # of its own, so structural inference would leave it untouched. A
        # context-declared float return seeds a +1 delta, which both makes the
        # FPU pass run and annotates the call as producing st(0) (fpret f0).
        self.arch.context_fpu_call_deltas = {"_returns_float": 1}
        out = self.infer(
            """
            CALL _returns_float, 0x0, 0x0
            RET
            """
        )
        self.assertEqual(out[0], "call _returns_float, 0x0, 0x0, 0x0, -0x1")

    def test_context_float_return_before_fstp(self) -> None:
        # A context-known float-returning call whose result is stored: the
        # seeded +1 delta annotates the call and the following fstp pops f0.
        self.arch.context_fpu_call_deltas = {"_returns_float": 1}
        out = self.infer(
            """
            CALL _returns_float, 0x0, 0x0
            FSTP dword ptr [_g]
            RET
            """
        )
        self.assertEqual(out[0], "call _returns_float, 0x0, 0x0, 0x0, -0x1")
        self.assertEqual(out[1], "fstp [_g], $f0")

    def test_no_fpu_no_context_delta_skips_pass(self) -> None:
        # A plain integer wrapper with no x87 and no context float delta is
        # left completely untouched (the cheap early-out still holds).
        from m2c.asm_file import AsmData
        from m2c.asm_pattern import AsmMatcher
        from m2c.x86_fpu import X86FpuRewritePattern

        body, labels = self._build_body(
            """
            CALL _plain, 0x0, 0x0
            RET
            """
        )
        matcher = AsmMatcher(self.arch, AsmData(), body, labels)
        self.assertIsNone(X86FpuRewritePattern().match(matcher))

    def test_infer_float_return_call(self) -> None:
        # Without a seed, the underflow at `fadd` (depth 0) is attributed to
        # the preceding call, whose delta is inferred as +1.
        out = self.infer(
            """
            CALL _foo, 0x0, 0x0
            FADD dword ptr [ESP + 0x4]
            FSTP dword ptr [_g]
            RET
            """
        )
        self.assertEqual(out[0], "call _foo, 0x0, 0x0, 0x0, -0x1")

    def test_infer_stack_consuming_helper(self) -> None:
        # A loop that fld+call each iteration grows the stack unless the call
        # consumes st0; inference finds delta -1 for the helper.
        out = self.infer(
            """
            L:
            FLD dword ptr [ESP + 0x4]
            CALL _ftol, 0x0, 0x0
            MOV EAX, dword ptr [ESP + 0x8]
            TEST EAX, EAX
            JNZ L
            RET
            """
        )
        self.assertEqual(out[1], "call _ftol, 0x0, 0x0, -0x1, 0x0")

    def test_float_argument_store(self) -> None:
        # `fstp [esp]` into a call's argument window becomes a subroutine-arg
        # store, not a plain memory store.
        out = self.rewrite(
            """
            FLD dword ptr [ESP + 0x4]
            PUSH ECX
            FSTP dword ptr [ESP]
            CALL _foo, 0x0, 0x4
            RET
            """
        )
        self.assertEqual(out[2], "fstparg 0x0, $f0")

    def test_transcendental_pop(self) -> None:
        # fpatan consumes st0 (depth 2 -> 1); operands are passed as (st0, st1).
        out = self.rewrite(
            """
            FLD dword ptr [ESP + 0x4]
            FLD dword ptr [ESP + 0x8]
            FPATAN
            FSTP dword ptr [_g]
            RET
            """
        )
        self.assertEqual(out[2], "fpatan $f1, $f0")
        # After the pop the result lives in f0 and is stored.
        self.assertEqual(out[3], "fstp [_g], $f0")

    def test_constant_load(self) -> None:
        out = self.rewrite(
            """
            FLD1
            FADD dword ptr [ESP + 0x4]
            FSTP dword ptr [_g]
            RET
            """
        )
        self.assertEqual(out[0], "fld1 $f0")

    def test_unsupported_instruction_raises(self) -> None:
        with self.assertRaises(DecompFailure) as cm:
            self.rewrite(
                """
                FLD dword ptr [ESP + 0x4]
                FPTAN
                RET
                """
            )
        self.assertIn("fptan", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
