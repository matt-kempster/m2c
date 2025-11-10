from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Set, TYPE_CHECKING

from ..asm_file import AsmSymbolicData, Label
from ..asm_instruction import (
    ARM_BARREL_SHIFTER_OPS,
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    BinOp,
    JumpTarget,
    Register,
    RegisterList,
    Writeback,
)
from ..asm_pattern import (
    AsmMatch,
    AsmMatcher,
    AsmPattern,
    Replacement,
    SimpleAsmPattern,
    make_pattern,
)
from ..error import DecompFailure
from ..instruction import Instruction, Location
from .common import CC_REGS, Cc, negate_cond, factor_cond, parse_suffix

if TYPE_CHECKING:  # pragma: no cover
    from .arch import ArmArch


MAGIC_FUNCTIONS = {
    "__divsi3": ("sdiv", "001"),
    "_idiv": ("sdiv", "001"),
    "__udivsi3": ("udiv", "001"),
    "_uidiv": ("udiv", "001"),
    "__modsi3": ("smod.fictive", "001"),
    "__umodsi3": ("umod.fictive", "001"),
    "_s32_div_f": ("sdivmod.fictive", "01"),
    "_idivmod": ("sdivmod.fictive", "01"),
    "_u32_div_f": ("udivmod.fictive", "01"),
    "_uidivmod": ("udivmod.fictive", "01"),
    "__clzsi2": ("clz", "00"),
    # Soft float emulation from fplib, see:
    # https://developer.arm.com/documentation/dui0475/m/floating-point-support/the-software-floating-point-library--fplib
    "_fadd": ("add.s.fictive", "001"),
    "_fsub": ("sub.s.fictive", "001"),
    "_frsub": ("sub.s.fictive", "010"),
    "_fmul": ("mul.s.fictive", "001"),
    "_fdiv": ("div.s.fictive", "001"),
    "_frdiv": ("div.s.fictive", "010"),
    "_fsqrt": ("sqrt.s.fictive", "00"),
    "_frnd": ("round.s.fictive", "00"),
    "_fflt": ("cvt.s.w.fictive", "00"),
    "_ffltu": ("cvt.s.u.fictive", "00"),
    "_ffix": ("cvt.w.s.fictive", "00"),
    "_ffixu": ("cvt.u.s.fictive", "00"),
    "_f2d": ("cvt.d.s.fictive", "00"),
    "_fls": ("c.lt.s.fictive", "01"),
    "_fgr": ("c.gt.s.fictive", "01"),
    "_fleq": ("c.le.s.fictive", "01"),
    "_fgeq": ("c.ge.s.fictive", "01"),
    "_feq": ("c.eq.s.fictive", "01"),
    "_fneq": ("c.neq.s.fictive", "01"),
    "_dadd": ("add.d.fictive", "002"),
    "_dsub": ("sub.d.fictive", "002"),
    "_drsub": ("sub.d.fictive", "020"),
    "_dmul": ("mul.d.fictive", "002"),
    "_ddiv": ("div.d.fictive", "002"),
    "_drdiv": ("div.d.fictive", "020"),
    "_dsqrt": ("sqrt.d.fictive", "00"),
    "_drnd": ("round.d.fictive", "00"),
    "_dflt": ("cvt.d.w.fictive", "00"),
    "_dfltu": ("cvt.d.u.fictive", "00"),
    "_dfix": ("cvt.w.d.fictive", "00"),
    "_dfixu": ("cvt.u.d.fictive", "00"),
    "_d2f": ("cvt.s.d.fictive", "00"),
    "_dls": ("c.lt.d.fictive", "02"),
    "_dgr": ("c.gt.d.fictive", "02"),
    "_dleq": ("c.le.d.fictive", "02"),
    "_dgeq": ("c.ge.d.fictive", "02"),
    "_deq": ("c.eq.d.fictive", "02"),
    "_dneq": ("c.neq.d.fictive", "02"),
}


@dataclass
class InstructionEffects:
    inputs: Set[Location] = field(default_factory=set)
    outputs: Set[Location] = field(default_factory=set)
    is_effectful: bool = False
    has_load: bool = False

    def can_move_before(self, instr: Instruction) -> bool:
        outputs = instr.outputs + instr.clobbers
        if set(instr.inputs + outputs) & self.outputs:
            return False
        if set(outputs) & self.inputs:
            return False
        if self.is_effectful and (instr.is_effectful or instr.is_load):
            return False
        if self.has_load and instr.is_effectful:
            return False
        return True

    def add(self, instr: Instruction) -> None:
        self.outputs |= set(instr.outputs + instr.clobbers)
        self.inputs |= set(instr.inputs)
        if instr.is_effectful:
            self.is_effectful = True
        if instr.is_load:
            self.has_load = True


class AddPcPcPattern(AsmPattern):
    """Detect switch jumps of the form "add pc, pc, reg, lsl 2"."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        instr = matcher.input[matcher.index]
        if not isinstance(instr, Instruction) or not instr.mnemonic.startswith("add"):
            return None
        base, cc, set_flags, direction = parse_suffix(instr.mnemonic)
        if (
            base != "add"
            or set_flags
            or instr.args[0] != Register("pc")
            or instr.args[1] != Register("pc")
            or not isinstance(instr.args[2], BinOp)
            or instr.args[2].op != "lsl"
            or instr.args[2].rhs != AsmLiteral(2)
        ):
            return None

        i = matcher.index + 1
        seen_first = False
        new_args: List[Argument] = [instr.args[2].lhs]
        while i < len(matcher.input):
            ins2 = matcher.input[i]
            i += 1
            if isinstance(ins2, Label):
                continue
            if not seen_first:
                seen_first = True
                continue
            if (
                ins2.mnemonic == "pop"
                and isinstance(ins2.args[0], RegisterList)
                and Register("pc") in ins2.args[0].regs
            ):
                new_args.append(AsmGlobalSymbol("_m2c_ret"))
            elif ins2.mnemonic == "b":
                assert isinstance(ins2.args[0], AsmGlobalSymbol)
                new_args.append(ins2.args[0])
            else:
                break
        cc_str = cc.value if cc is not None else ""
        jump_ins = AsmInstruction("tablejmp.fictive" + cc_str, new_args)
        return Replacement([jump_ins], 1)


class ShortJumpTablePattern(AsmPattern):
    """Detect switch jumps emitted by MWCC for Thumb."""

    pattern = make_pattern(
        "adds $x, $x, $x",
        "add $x, $x, $pc",
        "ldrh $x, [$x, 6]",
        "movs $x, $x, lsl 0x10",
        "movs $x, $x, asr 0x10",
        "add $pc, $pc, $x",
        ".label:",
    )

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        m = matcher.try_match(self.pattern)
        if not m:
            return None
        label = m.labels[".label"].names[0]
        ent = matcher.asm_data.values.get(label)
        if ent is None:
            return None
        new_args: List[Argument] = [m.regs["x"]]
        for data in ent.data:
            if (
                not isinstance(data, AsmSymbolicData)
                or not isinstance(data.data, BinOp)
                or data.data.rhs != AsmLiteral(2)
                or not isinstance(data.data.lhs, BinOp)
                or data.data.lhs.rhs != AsmGlobalSymbol(label)
            ):
                break
            new_args.append(data.data.lhs.lhs)
        jump_ins = AsmInstruction("tablejmp.fictive", new_args)
        return Replacement([jump_ins], len(self.pattern) - 1)


class ConditionalInstrPattern(AsmPattern):
    """Replace conditionally executed instructions by branches."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        matched_cc: Optional[Cc] = None
        i = 0
        if_instrs: List[AsmInstruction] = []
        else_instrs: List[AsmInstruction] = []
        before_instrs: List[Instruction] = []
        after_instrs: List[Instruction] = []
        mid_set = InstructionEffects()
        after_set = InstructionEffects()
        moved_before_streak = 0
        moved_after_streak = 0
        label1 = f"._m2c_cc_{matcher.index}"
        label2 = f"._m2c_cc2_{matcher.index}"
        flag_check: Optional[AsmInstruction] = None
        while matcher.index + i < len(matcher.input):
            instr = matcher.input[matcher.index + i]
            if not isinstance(instr, Instruction):
                break
            base, cc, set_flags, direction = parse_suffix(instr.mnemonic)
            if cc is None:
                if matched_cc is None:
                    return None
                if instr.is_jump():
                    break
                outputs = instr.outputs + instr.clobbers
                if mid_set.can_move_before(instr) and not after_instrs:
                    before_instrs.append(instr)
                    moved_before_streak += 1
                else:
                    after_set.add(instr)
                    after_instrs.append(instr)
                    moved_after_streak += 1
                i += 1
                continue
            outputs = instr.outputs + instr.clobbers
            if not after_set.can_move_before(instr):
                break
            if after_instrs and instr.is_jump():
                break
            mid_set.add(instr)
            new_instr = AsmInstruction(base + set_flags + direction, instr.args)
            if matched_cc is None:
                if base == "b":
                    break
                matched_cc = cc
            if cc != matched_cc and cc != negate_cond(matched_cc):
                break
            if flag_check is not None:
                if_instrs.append(flag_check)
            flag_check = None
            if matched_cc == cc:
                if_instrs.append(new_instr)
            else:
                else_instrs.append(new_instr)
            i += 1
            moved_before_streak = 0
            moved_after_streak = 0
            if instr.is_jump():
                break
            if CC_REGS[factor_cond(matched_cc)[0]] in outputs:
                if matched_cc == cc and not else_instrs:
                    b_mn = "b" + negate_cond(matched_cc).value
                    flag_check = AsmInstruction(b_mn, [AsmGlobalSymbol(label1)])
                else:
                    break
        if matched_cc is None:
            return None

        i -= moved_before_streak + moved_after_streak
        del before_instrs[len(before_instrs) - moved_before_streak :]
        del after_instrs[len(after_instrs) - moved_after_streak :]

        b_mn = "b" + negate_cond(matched_cc).value
        if else_instrs:
            return Replacement(
                [
                    *before_instrs,
                    AsmInstruction(b_mn, [AsmGlobalSymbol(label1)]),
                    *if_instrs,
                    AsmInstruction("b", [AsmGlobalSymbol(label2)]),
                    Label([label1]),
                    *else_instrs,
                    Label([label2]),
                    *after_instrs,
                ],
                i,
            )
        else:
            return Replacement(
                [
                    *before_instrs,
                    AsmInstruction(b_mn, [AsmGlobalSymbol(label1)]),
                    *if_instrs,
                    Label([label1]),
                    *after_instrs,
                ],
                i,
            )


class NegatedRegAddrModePattern(AsmPattern):
    """Replace negated registers in address modes by neg instructions."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        instr = matcher.input[matcher.index]
        if not isinstance(instr, Instruction):
            return None
        if not instr.args or not isinstance(instr.args[-1], AsmAddressMode):
            return None
        addend = instr.args[-1].addend
        if isinstance(addend, BinOp) and addend.op != "-":
            inner = addend.lhs
        else:
            inner = addend
        if not isinstance(inner, BinOp) or inner.op != "-":
            return None
        temp = Register.fictive("neg")
        new_addend: Argument
        if isinstance(addend, BinOp) and addend.op != "-":
            new_addend = replace(addend, lhs=temp)
        else:
            new_addend = temp
        new_args = list(instr.args)
        new_args[-1] = replace(instr.args[-1], addend=new_addend)
        return Replacement(
            [
                AsmInstruction("rsb", [temp, inner.rhs, AsmLiteral(0)]),
                AsmInstruction(instr.mnemonic, new_args),
            ],
            1,
        )


class ShiftedRegAddrModePattern(AsmPattern):
    """Replace barrel shifted registers in address modes."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        instr = matcher.input[matcher.index]
        if not isinstance(instr, Instruction):
            return None
        if not instr.args or not isinstance(instr.args[-1], AsmAddressMode):
            return None
        addend = instr.args[-1].addend
        if not isinstance(addend, BinOp) or addend.op not in ARM_BARREL_SHIFTER_OPS:
            return None
        temp = Register.fictive(addend.op)
        new_args = list(instr.args)
        new_args[-1] = replace(instr.args[-1], addend=temp)
        return Replacement(
            [
                AsmInstruction("mov", [temp, addend]),
                AsmInstruction(instr.mnemonic, new_args),
            ],
            1,
        )


class AddrModeWritebackPattern(AsmPattern):
    """Replace writebacks in address modes by additional instructions."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        instr = matcher.input[matcher.index]
        if not isinstance(instr, Instruction) or not instr.args:
            return None
        addr = instr.args[-1]
        if not isinstance(addr, AsmAddressMode) or addr.writeback is None:
            return None
        new_args = list(instr.args)
        new_args[-1] = AsmAddressMode(addr.base, AsmLiteral(0), None)
        if instr.args[0] == addr.base:
            raise DecompFailure(
                "Writeback of same register as is being loaded/stored, "
                f"for instruction: {instr}"
            )
        if addr.writeback == Writeback.PRE:
            return Replacement(
                [
                    AsmInstruction("add", [addr.base, addr.base, addr.addend]),
                    AsmInstruction(instr.mnemonic, new_args),
                ],
                1,
            )
        else:
            return Replacement(
                [
                    AsmInstruction(instr.mnemonic, new_args),
                    AsmInstruction("add", [addr.base, addr.base, addr.addend]),
                ],
                1,
            )


class RegRegAddrModePattern(AsmPattern):
    """Replace register addends in address modes by a separate add."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        instr = matcher.input[matcher.index]
        if not isinstance(instr, Instruction) or not instr.args:
            return None
        arg = instr.args[-1]
        if not isinstance(arg, AsmAddressMode) or not isinstance(arg.addend, Register):
            return None
        assert arg.writeback is None
        temp = Register.fictive("mem_loc")
        new_args = list(instr.args)
        new_args[-1] = AsmAddressMode(temp, AsmLiteral(0), None)
        return Replacement(
            [
                AsmInstruction("add", [temp, arg.base, arg.addend]),
                AsmInstruction(instr.mnemonic, new_args),
            ],
            1,
        )


class ShiftedRegPattern(AsmPattern):
    """Replace barrel shifted registers by additional instructions."""

    @staticmethod
    def _sets_flags(base: str, set_flags: str) -> bool:
        return bool(
            base in ("tst", "teq")
            or (base in ("mov", "mvn", "and", "eor", "orr", "orn", "bic") and set_flags)
        )

    @staticmethod
    def _get_literal_carry(base: str, arg: Argument) -> Optional[int]:
        if not isinstance(arg, AsmLiteral):
            return None
        value = arg.value & 0xFFFFFFFF
        if value >= 0x100 and value < (value & -value) * 0x100:
            return value >> 31
        if base in ("mov", "mvn"):
            value ^= 0xFFFFFFFF
        if value >= 0x100 and value < (value & -value) * 0x100:
            return value >> 31
        return None

    @classmethod
    def sets_flags_based_on_barrel_shifter(
        cls, base: str, set_flags: str, args: List[Argument]
    ) -> bool:
        if not args:
            return False
        arg = args[-1]
        return cls._sets_flags(base, set_flags) and (
            cls._get_literal_carry(base, arg) is not None
            or (isinstance(arg, BinOp) and arg.op in ARM_BARREL_SHIFTER_OPS)
        )

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        instr = matcher.input[matcher.index]
        if not isinstance(instr, Instruction) or not instr.args:
            return None
        base, _, set_flags, _ = parse_suffix(instr.mnemonic)
        arg = instr.args[-1]

        literal_carry = self._get_literal_carry(base, arg)
        if literal_carry is not None:
            if not self._sets_flags(base, set_flags):
                return None
            return Replacement(
                [
                    instr,
                    AsmInstruction("setcarryi.fictive", [AsmLiteral(literal_carry)]),
                ],
                1,
            )

        if not isinstance(arg, BinOp) or arg.op not in ARM_BARREL_SHIFTER_OPS:
            return None

        temp = Register.fictive(arg.op)
        shift_ins = AsmInstruction(arg.op, [temp, arg.lhs, arg.rhs])
        if arg.op == "rrx":
            shift_ins.args.pop()
        new_instrs = [
            shift_ins,
            AsmInstruction(instr.mnemonic, instr.args[:-1] + [temp]),
        ]

        if self._sets_flags(base, set_flags):
            new_instrs.append(AsmInstruction("setcarry.fictive", [temp]))

        return Replacement(new_instrs, 1)


class PopAndReturnPattern(SimpleAsmPattern):
    pattern = make_pattern(
        "pop {x}",
        "add sp, sp, 0x10?",
        "bx $x",
    )

    def replace(self, m: AsmMatch) -> Replacement:
        return Replacement([AsmInstruction("bx", [Register("lr")])], len(m.body))


class TailCallPattern(SimpleAsmPattern):
    pattern = make_pattern("bx $x")

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        if m.regs["x"] == Register("lr"):
            return None
        return Replacement(
            [
                AsmInstruction("blx", [m.regs["x"]]),
                AsmInstruction("bx", [Register("lr")]),
            ],
            len(m.body),
        )


class BlBranchPattern(AsmPattern):
    """Replace bl by b when used for intra-function jumps."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        def is_local_bl(instr: Instruction) -> bool:
            return (
                instr.mnemonic == "bl"
                and isinstance(instr.args[0], AsmGlobalSymbol)
                and matcher.is_local_label(instr.args[0].symbol_name)
            )

        instr = matcher.input[matcher.index]
        if isinstance(instr, Instruction) and is_local_bl(instr):
            return Replacement(
                [
                    AsmInstruction("b", instr.args),
                ],
                1,
                clobbers=[Register("lr")],
            )

        if matcher.index + 2 >= len(matcher.input):
            return None
        instr2 = matcher.input[matcher.index + 1]
        label = matcher.input[matcher.index + 2]
        if (
            isinstance(instr, Instruction)
            and isinstance(instr2, Instruction)
            and isinstance(label, Label)
            and is_local_bl(instr2)
            and isinstance(instr.jump_target, JumpTarget)
            and instr.jump_target.target in label.names
            and instr.is_conditional
        ):
            base, cc, _, _ = parse_suffix(instr.mnemonic)
            if base == "b" and cc is not None:
                b_mn = "b" + negate_cond(cc).value
                return Replacement(
                    [
                        AsmInstruction(b_mn, instr2.args),
                    ],
                    2,
                    clobbers=[Register("lr")],
                )
        return None


class MagicFuncPattern(SimpleAsmPattern):
    pattern = make_pattern("bl")

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        ins = m.body[0]
        assert isinstance(ins, Instruction)
        assert isinstance(ins.args[0], AsmGlobalSymbol)
        target = ins.args[0].symbol_name
        if target.startswith("_call_via_"):
            from .arch import ArmArch

            reg_name = target[10:]
            reg = ArmArch.aliased_regs.get(reg_name)
            if reg is None and Register(reg_name) in ArmArch.all_regs:
                reg = Register(reg_name)
            if reg is not None:
                return Replacement([AsmInstruction("blx", [reg])], 1)
        if target.startswith("__aeabi_"):
            target = target[7:]
        if target in MAGIC_FUNCTIONS:
            mn, arg_str = MAGIC_FUNCTIONS[target]
            args: List[Argument] = [Register("r" + x) for x in arg_str]
            return Replacement([AsmInstruction(mn, args)], 1)
        return None


__all__ = [
    "AddPcPcPattern",
    "AddrModeWritebackPattern",
    "BlBranchPattern",
    "ConditionalInstrPattern",
    "InstructionEffects",
    "MagicFuncPattern",
    "NegatedRegAddrModePattern",
    "PopAndReturnPattern",
    "RegRegAddrModePattern",
    "ShiftedRegAddrModePattern",
    "ShiftedRegPattern",
    "ShortJumpTablePattern",
    "TailCallPattern",
]
