from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from .error import DecompFailure
from .options import Target
from .asm_file import AsmSymbolicData, Label
from .asm_instruction import (
    ARM_BARREL_SHIFTER_OPS,
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    AsmState,
    BinOp,
    JumpTarget,
    Macro,
    Register,
    RegisterList,
    Writeback,
    get_jump_target,
)
from .asm_pattern import (
    AsmMatch,
    AsmMatcher,
    AsmPattern,
    Replacement,
    ReplacementPart,
    SimpleAsmPattern,
    make_pattern,
)
from .instruction import (
    Instruction,
    InstructionMeta,
    Location,
    StackLocation,
)
from .translate import (
    Abi,
    AbiArgSlot,
    AddressMode,
    ArgLoc,
    Arch,
    BinaryOp,
    CarryBit,
    Cast,
    Condition,
    ErrorExpr,
    ExprStmt,
    Expression,
    InstrArgs,
    InstrMap,
    Literal,
    NodeState,
    SecondF64Half,
    StmtInstrMap,
    StoreInstrMap,
    UnaryOp,
    as_intish,
    as_type,
    as_uintish,
    as_f32,
    as_f64,
    format_hex,
)
from .evaluate import (
    condition_from_expr,
    deref,
    error_stmt,
    eval_arm_cmp,
    fn_op,
    fold_divmod,
    fold_mul_chains,
    handle_add,
    handle_add_arm,
    handle_add_double,
    handle_add_float,
    handle_add_real,
    handle_addi,
    handle_arm_mov,
    handle_bitinv,
    handle_convert,
    handle_la,
    handle_load,
    handle_or,
    handle_shift_right,
    handle_sll,
    handle_sub,
    handle_sub_arm,
    make_store,
    make_store_real,
    replace_bitand,
    set_arm_flags_from_add,
    void_fn_op,
)
from .types import FunctionSignature, Type


SUFFIXABLE_INSTRUCTIONS: Set[str] = {
    "adc",
    "adr",
    "add",
    "and",
    "asr",
    "b",
    "bic",
    "bl",
    "blx",
    "bx",
    "cbnz",
    "cbz",
    "clz",
    "cmn",
    "cmp",
    "cpy",
    "eor",
    "ldm",
    "ldr",
    "ldrb",
    "ldrh",
    "ldrsb",
    "ldrsh",
    "lsl",
    "lsr",
    "mla",
    "mls",
    "movimm.fictive",
    "mov",
    "mul",
    "mvn",
    "neg",
    "nop",
    "orn",
    "orr",
    "pop",
    "push",
    "rbit",
    "rev",
    "rev16",
    "revsh",
    "ror",
    "rrx",
    "rsb",
    "rsc",
    "sbc",
    "sdiv",
    "smlabb",
    "smlal",
    "smmla",
    "smmls",
    "smmul",
    "smulbb",
    "smull",
    "stm",
    "str",
    "strb",
    "strh",
    "sub",
    "tablejmp.fictive",
    "teq",
    "tst",
    "udiv",
    "umlal",
    "umull",
}

LENGTH_THREE: Set[str] = {
    "adc",
    "add",
    "and",
    "asr",
    "bic",
    "eor",
    "lsl",
    "lsr",
    "mul",
    "orn",
    "orr",
    "ror",
    "rsb",
    "rsc",
    "sbc",
    "sub",
}

THUMB1_FLAG_SETTING: Set[str] = {
    "adc",
    "add",
    "and",
    "asr",
    "bic",
    "eor",
    "lsl",
    "lsr",
    "mov",
    "mul",
    "mvn",
    "neg",
    "orr",
    "ror",
    "sbc",
    "sub",
}


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


HI_REGS: Set[Register] = {
    Register("r8"),
    Register("r9"),
    Register("r10"),
    Register("r11"),
    Register("r12"),
    Register("sp"),
    Register("lr"),
    Register("pc"),
}


class Cc(Enum):
    EQ = "eq"
    NE = "ne"
    CS = "cs"
    CC = "cc"
    MI = "mi"
    PL = "pl"
    VS = "vs"
    VC = "vc"
    HI = "hi"
    LS = "ls"
    GE = "ge"
    LT = "lt"
    GT = "gt"
    LE = "le"
    AL = "al"


CC_REGS: Dict[Cc, Register] = {
    Cc.EQ: Register("z"),
    Cc.CS: Register("c"),
    Cc.MI: Register("n"),
    Cc.VS: Register("v"),
    Cc.HI: Register("hi"),
    Cc.GE: Register("ge"),
    Cc.GT: Register("gt"),
}


def negate_cond(cc: Cc) -> Cc:
    return {
        Cc.EQ: Cc.NE,
        Cc.NE: Cc.EQ,
        Cc.CS: Cc.CC,
        Cc.CC: Cc.CS,
        Cc.MI: Cc.PL,
        Cc.PL: Cc.MI,
        Cc.VS: Cc.VC,
        Cc.VC: Cc.VS,
        Cc.HI: Cc.LS,
        Cc.LS: Cc.HI,
        Cc.GE: Cc.LT,
        Cc.LT: Cc.GE,
        Cc.GT: Cc.LE,
        Cc.LE: Cc.GT,
    }[cc]


def factor_cond(cc: Cc) -> Tuple[Cc, bool]:
    return {
        Cc.EQ: (Cc.EQ, False),
        Cc.NE: (Cc.EQ, True),
        Cc.CS: (Cc.CS, False),
        Cc.CC: (Cc.CS, True),
        Cc.MI: (Cc.MI, False),
        Cc.PL: (Cc.MI, True),
        Cc.VS: (Cc.VS, False),
        Cc.VC: (Cc.VS, True),
        Cc.HI: (Cc.HI, False),
        Cc.LS: (Cc.HI, True),
        Cc.GE: (Cc.GE, False),
        Cc.LT: (Cc.GE, True),
        Cc.GT: (Cc.GT, False),
        Cc.LE: (Cc.GT, True),
    }[cc]


def parse_suffix(mnemonic: str) -> Tuple[str, Optional[Cc], str, str]:
    ldm = mnemonic.startswith("ldm")
    stm = mnemonic.startswith("stm")

    def strip_cc(mnemonic: str) -> Tuple[str, Optional[Cc]]:
        for suffix in [cond.value for cond in Cc] + ["hs", "lo"]:
            if mnemonic.endswith(suffix):
                if suffix == "hs":
                    cc = Cc.CS
                elif suffix == "lo":
                    cc = Cc.CC
                else:
                    cc = Cc(suffix)
                return mnemonic[: -len(suffix)], cc
        return mnemonic, None

    def strip_dir(mnemonic: str) -> Tuple[str, str]:
        if not ldm and not stm:
            return mnemonic, ""
        if any(mnemonic.endswith(suffix) for suffix in ("ia", "ib", "da", "db")):
            return mnemonic[:-2], mnemonic[-2:]
        if any(mnemonic.endswith(suffix) for suffix in ("fa", "ea", "fd", "ed")):
            # Pre-UAL syntax
            tr = {
                "fa": ["da", "ib"],
                "ea": ["db", "ia"],
                "fd": ["ia", "db"],
                "ed": ["ib", "da"],
            }
            return mnemonic[:-2], tr[mnemonic[-2:]][0 if ldm else 1]
        return mnemonic, ""

    # bls should be parsed as b + ls, not bl + s
    s_ok = not mnemonic.startswith("b") or mnemonic.startswith("bic")

    # Strip memory size from the end (legacy ARM syntax). We re-attach it
    # later and treat it as part of the mnemonic.
    memsize = ""
    if mnemonic.startswith("str") or mnemonic.startswith("ldr"):
        # ldrhs should be parsed as ldr + hs, not ldrh + s
        s_ok = False
        for suffix in ("b", "h", "d"):
            if mnemonic.endswith(suffix):
                mnemonic = mnemonic[:-1]
                memsize = suffix
                break
        if (
            memsize in ("b", "h")
            and mnemonic.endswith("s")
            and (not strip_cc(mnemonic)[1] or strip_cc(mnemonic[:-1])[1])
        ):
            mnemonic = mnemonic[:-1]
            memsize = "s" + memsize

    # We want to support both UAL and legacy ARM syntaxes, which means condition
    # codes can appear in multiple places in the mnemonic. For example, both
    # lsllss and lslsls are valid. We deal with this by trying both possibilities,
    # and returning once we find an instruction that we recognize. Having a list
    # of recognized instructions is somewhat unavoidable, in order to avoid
    # ambiguities in cases like sdivcs (sdiv + cs) vs addvcs (add + vc + s).
    orig_mn = mnemonic
    for cc_is_last in (False, True):
        mnemonic, direction = strip_dir(orig_mn)
        cc: Optional[Cc] = None
        set_flags = ""
        if cc_is_last:
            mnemonic, cc = strip_cc(mnemonic)
            if not direction:
                mnemonic, direction = strip_dir(mnemonic)
        if mnemonic in SUFFIXABLE_INSTRUCTIONS:
            return mnemonic + memsize, cc, set_flags, direction
        if mnemonic.endswith("s") and s_ok:
            set_flags = "s"
            mnemonic = mnemonic[:-1]
            if mnemonic in SUFFIXABLE_INSTRUCTIONS:
                return mnemonic + memsize, cc, set_flags, direction
        if not cc_is_last:
            mnemonic, cc = strip_cc(mnemonic)
            if mnemonic in SUFFIXABLE_INSTRUCTIONS:
                return mnemonic + memsize, cc, set_flags, direction

    return orig_mn + memsize, None, "", ""


def get_ldm_stm_offset(i: int, num_regs: int, direction: str) -> int:
    if direction == "ia":
        return 4 * i
    if direction == "ib":
        return 4 * (i + 1)
    if direction == "da":
        return 4 * (i + 1 - num_regs)
    if direction == "db":
        return 4 * (i - num_regs)
    assert False, f"bad ldm/stm direction {direction}"


def other_f64_reg(reg: Register) -> Register:
    num = int(reg.register_name[1:])
    return Register(f"r{num ^ 1}")


class AddPcPcPattern(AsmPattern):
    """Detect switch jumps of the form "add pc, pc, reg, lsl 2" and replace
    them by instructions that reference the jump table."""

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
                # Fallback instruction if cc condition is false
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
                    # Only match if the first instruction is conditional.
                    return None
                if instr.is_jump():
                    break
                # This instruction does not belong in the if, but we can move
                # it to the front or end of it, assuming that we don't reorder
                # impure instructions and there are no dependency issues.
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
                # If we have a jump, there is no way to put instructions after.
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
                    # Flags have been updated, so we need to emit a check for
                    # whether we should branch to the else block (label1).
                    # Skip this if this is the last instruction in the block.
                    # We could alternatively just break here, but handling this
                    # makes us detect cmpeq chains as chains of &&.
                    b_mn = "b" + negate_cond(matched_cc).value
                    flag_check = AsmInstruction(b_mn, [AsmGlobalSymbol(label1)])
                else:
                    # Jumping into the middle of the other block is not worth
                    # the complexity to support (and might introduce irreducible
                    # control flow?).
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
    """Replace barrel shifted registers in address modes by additional instructions."""

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
        # Load/store instructions which specify writeback of the base register
        # are UNPREDICTABLE if the base register to be written back matches
        # the register to be loaded/stored (Rn == Rt).
        # In the remaining cases, this rewrite is correct.
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
    """Replace register addends in address modes by a separate add instruction."""

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
        # Carry = bit 31 if the constant is greater than 255 and can be
        # produced by shifting an 8-bit value
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
        base, cc, set_flags, direction = parse_suffix(instr.mnemonic)
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
    pattern = make_pattern(
        "bx $x",
    )

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
    """Replace bl by b when used for intra-function jumps, which is common
    in Thumb code to extend the range of branches."""

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

        # bcc .skip
        # bl .target
        # .skip:
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
            reg_name = target[10:]
            reg = ArmArch.aliased_regs.get(reg_name)
            if reg is None and Register(reg_name) in ArmArch.all_regs:
                reg = Register(reg_name)
            if reg is not None:
                return Replacement([AsmInstruction("blx", [reg])], 1)
        if target.startswith("__aeabi_"):
            # Both __aeabi_fadd and _fadd occur in practice; let's strip all
            # __aeabi prefixes to be more accepting.
            target = target[7:]
        if target in MAGIC_FUNCTIONS:
            mn, arg_str = MAGIC_FUNCTIONS[target]
            args: List[Argument] = [Register("r" + x) for x in arg_str]
            return Replacement([AsmInstruction(mn, args)], 1)
        return None


class ArmArch(Arch):
    arch = Target.ArchEnum.ARM

    re_comment = r"^[ \t]*#.*|[@;].*"
    supports_dollar_regs = False

    home_space_size = 0

    stack_pointer_reg = Register("sp")
    frame_pointer_regs = [Register("r7"), Register("r11")]  # for Thumb/ARM respectively
    return_address_reg = Register("lr")

    base_return_regs = [(Register("r0"), False)]
    all_return_regs = [Register("r0"), Register("r1")]
    argument_regs = [Register(r) for r in ["r0", "r1", "r2", "r3"]]
    simple_temp_regs = [Register("r12")]
    flag_regs = [Register(r) for r in ["n", "z", "c", "v", "hi", "ge", "gt"]]
    temp_regs = argument_regs + simple_temp_regs + flag_regs
    saved_regs = [
        Register(r)
        for r in [
            "r4",
            "r5",
            "r6",
            "r7",
            "r8",
            "r9",
            "r10",
            "r11",
            "lr",
        ]
    ]
    all_regs = saved_regs + temp_regs + [stack_pointer_reg, Register("pc")]

    aliased_regs = {
        "a1": Register("r0"),
        "a2": Register("r1"),
        "a3": Register("r2"),
        "a4": Register("r3"),
        "v1": Register("r4"),
        "v2": Register("r5"),
        "v3": Register("r6"),
        "v4": Register("r7"),
        "v5": Register("r8"),
        "v6": Register("r9"),
        "v7": Register("r10"),
        "v8": Register("r11"),
        "wr": Register("r7"),
        "sb": Register("r9"),
        "tr": Register("r9"),
        "sl": Register("r10"),
        "fp": Register("r11"),
        "ip": Register("r12"),
        "r13": Register("sp"),
        "r14": Register("lr"),
        "r15": Register("pc"),
    }

    @classmethod
    def missing_return(cls) -> List[Instruction]:
        return [
            cls.parse("bx", [Register("lr")], InstructionMeta.missing()),
        ]

    @classmethod
    def normalize_instruction(
        cls, instr: AsmInstruction, asm_state: AsmState
    ) -> AsmInstruction:
        if instr.mnemonic.endswith(".n") or instr.mnemonic.endswith(".w"):
            instr = replace(instr, mnemonic=instr.mnemonic[:-2])
        if asm_state.is_thumb and not asm_state.is_unified:
            if len(instr.mnemonic) > 3 and instr.mnemonic.endswith("s"):
                # Give a best-effort warning for missing ".syntax unified" directives.
                # This isn't fool-proof, but unfortunately we need to default to
                # pre-UAL to match GNU as, and there are cases where semantics have
                # changed.
                raise DecompFailure("Missing '.syntax unified' marker")
            if instr.mnemonic in THUMB1_FLAG_SETTING and all(
                arg not in HI_REGS for arg in instr.args
            ):
                instr = replace(instr, mnemonic=instr.mnemonic + "s")
        return cls.normalize_instruction_real(instr)

    @classmethod
    def normalize_instruction_real(cls, instr: AsmInstruction) -> AsmInstruction:
        base, cc, set_flags, direction = parse_suffix(instr.mnemonic)
        if cc == Cc.AL:
            cc = None
            instr = replace(instr, mnemonic=base + set_flags + direction)
        cc_str = cc.value if cc else ""
        suffix = cc_str + set_flags + direction
        args = instr.args
        if len(args) == 3:
            if base == "lsl" and args[2] == AsmLiteral(0):
                return AsmInstruction("mov" + suffix, args[:2])
            if base == "add" and not set_flags and args[2] == AsmLiteral(0):
                return AsmInstruction("mov" + suffix, args[:2])
            if base in ("asr", "lsl", "lsr", "ror"):
                return AsmInstruction(
                    "mov" + suffix, [args[0], BinOp(base, args[1], args[2])]
                )
        if len(args) == 2:
            if instr.mnemonic == "mov" and args[0] == args[1] == Register("r8"):
                return AsmInstruction("nop", [])
            if (
                base == "ldr"
                and isinstance(args[0], Register)
                and isinstance(args[1], AsmAddressMode)
                and args[1].base == Register("sp")
                and args[1].addend == AsmLiteral(4)
                and args[1].writeback == Writeback.POST
            ):
                return AsmInstruction("pop" + suffix, [RegisterList([args[0]])])
            if (
                base == "ldr"
                and isinstance(args[1], Macro)
                and args[1].macro_name == "eqsign"
            ):
                if isinstance(args[1].argument, AsmLiteral):
                    mn = "movimm.fictive"
                else:
                    mn = "adr"
                return AsmInstruction(mn + suffix, [args[0], args[1].argument])
            if (
                base == "str"
                and isinstance(args[0], Register)
                and isinstance(args[1], AsmAddressMode)
                and args[1].base == Register("sp")
                and args[1].addend == AsmLiteral(-4)
                and args[1].writeback == Writeback.PRE
            ):
                return AsmInstruction("push" + suffix, [RegisterList([args[0]])])
            if base == "cpy":
                return AsmInstruction("mov" + suffix, args)
            if base == "neg":
                return AsmInstruction("rsb" + suffix, args + [AsmLiteral(0)])
            if base == "rrx":
                return AsmInstruction(
                    "mov" + suffix, [args[0], BinOp(base, args[1], AsmLiteral(1))]
                )
            if base in ("stm", "ldm") and not direction:
                return cls.normalize_instruction_real(
                    AsmInstruction(instr.mnemonic + "ia", args)
                )
            sp_excl = AsmAddressMode(Register("sp"), AsmLiteral(0), Writeback.PRE)
            if base == "stm" and direction == "db" and args[0] == sp_excl:
                return AsmInstruction("push" + cc_str, [args[1]])
            if base == "ldm" and direction == "ia" and args[0] == sp_excl:
                return AsmInstruction("pop" + cc_str, [args[1]])
            if base in LENGTH_THREE:
                return cls.normalize_instruction_real(
                    AsmInstruction(instr.mnemonic, [args[0]] + args)
                )
        if instr.mnemonic.startswith("it"):
            # Conditions are encoded in the following instructions already
            return AsmInstruction("nop", [])
        return instr

    @classmethod
    def parse(
        cls, mnemonic: str, args: List[Argument], meta: InstructionMeta
    ) -> Instruction:
        inputs: List[Location] = []
        clobbers: List[Location] = []
        outputs: List[Location] = []
        jump_target: Optional[Union[JumpTarget, Register, list[JumpTarget]]] = None
        function_target: Optional[Argument] = None
        is_conditional = False
        is_return = False
        is_load = False
        is_store = False
        is_effectful = True
        eval_fn: Optional[Callable[[NodeState, InstrArgs], object]] = None

        instr_str = str(AsmInstruction(mnemonic, args))

        base, cc, set_flags, direction = parse_suffix(mnemonic)
        if cc is not None:
            cc, cc_negated = factor_cond(cc)
        else:
            cc_negated = False

        def get_inputs(starti: int) -> List[Location]:
            ret: List[Location] = []
            for arg in args[starti:]:
                if isinstance(arg, AsmAddressMode):
                    ret.append(arg.base)
                    arg = arg.addend
                if isinstance(arg, BinOp):
                    if isinstance(arg.rhs, Register):
                        ret.append(arg.rhs)
                    arg = arg.lhs
                if (
                    isinstance(arg, BinOp)
                    and arg.op == "-"
                    and arg.lhs == AsmLiteral(0)
                ):
                    arg = arg.rhs
                if isinstance(arg, Register):
                    ret.append(arg)
            return ret

        if base == "b":
            # Conditional or unconditional branch
            assert len(args) == 1
            jump_target = get_jump_target(args[0])

            if cc is not None:
                is_conditional = True

                def eval_fn(s: NodeState, a: InstrArgs) -> None:
                    cond = condition_from_expr(a.regs[CC_REGS[cc]])
                    if cc_negated:
                        cond = cond.negated()
                    s.set_branch_condition(cond)

        elif base in ("cbz", "cbnz"):
            # Thumb conditional branch
            assert len(args) == 2
            assert isinstance(args[0], Register)
            inputs = [args[0]]
            jump_target = get_jump_target(args[1])
            is_conditional = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                op = "==" if base == "cbz" else "!="
                s.set_branch_condition(BinaryOp.icmp(a.reg(0), op, Literal(0)))

        elif base == "bx" and args[0] == Register("lr"):
            # Return
            assert len(args) == 1
            inputs = [Register("lr")]
            is_return = True
        elif base == "pop":
            assert len(args) == 1
            assert isinstance(args[0], RegisterList)
            outputs = list(args[0].regs)
            if Register("pc") in args[0].regs:
                is_return = True
            else:
                # Fow now, assume pop instructions are only used at the end of
                # the function, where they can be ignored for our purposes.
                pass
        elif base == "blx" and isinstance(args[0], Register):
            # Function call to pointer
            inputs = list(cls.argument_regs)
            inputs.append(args[0])
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            function_target = args[0]
            eval_fn = lambda s, a: s.make_function_call(a.reg(0), outputs)
        elif base in ("bl", "blx"):
            # Function call to label
            assert len(args) == 1
            inputs = list(cls.argument_regs)
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            function_target = args[0]
            eval_fn = lambda s, a: s.make_function_call(a.sym_imm(0), outputs)
        elif base == "ldr" and args[0] == Register("pc"):
            # Tail call to result of load, probably from literal pool.
            # We could resolve the target statically in that case but it's
            # not worth the effort given how seldom this comes up.
            assert not set_flags
            inputs = get_inputs(1)
            outputs = list(cls.all_return_regs)
            is_return = True
            function_target = args[1]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                target = handle_load(a, type=Type.reg32(likely_float=False))
                s.make_function_call(target, outputs)

        elif base == "tablejmp.fictive":
            # Switch tables of the form "add pc, pc, reg, lsl 2"
            assert len(args) >= 1 and isinstance(args[0], Register)
            targets = []
            for arg in args[1:]:
                assert isinstance(arg, AsmGlobalSymbol)
                targets.append(JumpTarget(arg.symbol_name))
            inputs = [args[0]]
            jump_target = targets
            is_conditional = True
            eval_fn = lambda s, a: s.set_switch_expr(a.reg(0), just_index=True)
        elif base in cls.instrs_no_flags:
            assert not set_flags
            assert isinstance(args[0], Register)
            outputs = [args[0]]
            inputs = get_inputs(1)
            is_effectful = False
            is_load = base.startswith("ldr")

            if base.startswith("cvt."):
                assert len(args) == 2 and isinstance(args[1], Register)
                mn_parts = base.split(".")
                if mn_parts[1] == "d":
                    outputs.append(other_f64_reg(args[0]))
                if mn_parts[2] == "d":
                    inputs.append(other_f64_reg(args[1]))
            elif base.endswith(".d.fictive"):
                for reg in args[1:]:
                    assert isinstance(reg, Register)
                    inputs.extend([reg, other_f64_reg(reg)])
                outputs.append(other_f64_reg(args[0]))

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                target = a.reg_ref(0)
                s.set_reg(target, cls.instrs_no_flags[base](a))
                if len(outputs) == 2:
                    s.set_reg(other_f64_reg(target), SecondF64Half())

        elif base in cls.instrs_divmod:
            assert not set_flags
            outputs = [Register("r0"), Register("r1")]
            inputs = get_inputs(0)
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                div, mod = cls.instrs_divmod[base](a)
                s.set_reg_real(Register("r0"), div, function_return=True)
                s.set_reg_real(Register("r1"), mod, function_return=True)

        elif base == "ldm":
            assert not set_flags
            assert direction in ("ia", "ib", "da", "db")
            assert isinstance(args[1], RegisterList)
            reg_list = args[1]
            outputs = list(args[1].regs)
            writeback = False
            if isinstance(args[0], Register):
                base_reg = args[0]
            else:
                assert isinstance(args[0], AsmAddressMode)
                assert args[0].addend == AsmLiteral(0)
                assert args[0].writeback == Writeback.PRE
                base_reg = args[0].base
                if base_reg in outputs:
                    # According to ARMv6 documentation, "If the base register
                    # <Rn> is specified in <registers>, and base register
                    # write-back is specified, the final value of <Rn> is
                    # UNPREDICTABLE." The pattern does come up in practice
                    # though, and my best guess at the behavior is that we
                    # should just ignore the writeback.
                    pass
                else:
                    outputs.append(base_reg)
                    writeback = True
            inputs = [base_reg]
            is_effectful = False
            is_load = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                loads: List[Expression] = []
                for i in range(len(reg_list.regs)):
                    offset = get_ldm_stm_offset(i, len(reg_list.regs), direction)
                    target = AddressMode(offset, base_reg)
                    loads.append(deref(target, a.regs, a.stack_info, size=4))
                for r, v in zip(reg_list.regs, loads):
                    s.set_reg(
                        r, as_type(v, Type.reg32(likely_float=False), silent=True)
                    )
                if writeback:
                    imm = Literal(4 * len(reg_list.regs))
                    op = "+" if direction in ("ia", "ib") else "-"
                    s.set_reg(base_reg, BinaryOp.intptr(a.regs[base_reg], op, imm))

        elif base == "stm":
            assert not set_flags
            assert direction in ("ia", "ib", "da", "db")
            assert isinstance(args[1], RegisterList)
            reg_list = args[1]
            inputs = list(args[1].regs)
            writeback = False
            if isinstance(args[0], Register):
                base_reg = args[0]
            else:
                assert isinstance(args[0], AsmAddressMode)
                assert args[0].addend == AsmLiteral(0)
                assert args[0].writeback == Writeback.PRE
                base_reg = args[0].base
                outputs.append(base_reg)
                writeback = True
            inputs.append(base_reg)
            is_store = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                for i, r in enumerate(reg_list.regs):
                    offset = get_ldm_stm_offset(i, len(reg_list.regs), direction)
                    target = AddressMode(offset, base_reg)
                    store = make_store_real(
                        a.regs[r],
                        a.regs.get_raw(r),
                        target,
                        a.regs,
                        a.stack_info,
                        Type.reg32(likely_float=False),
                    )
                    if store is not None:
                        s.store_memory(store, r)
                if writeback:
                    imm = Literal(4 * len(reg_list.regs))
                    op = "+" if direction in ("ia", "ib") else "-"
                    s.set_reg(base_reg, BinaryOp.intptr(a.regs[base_reg], op, imm))

        elif base == "mov" and args[0] == Register("pc"):
            assert len(args) == 2 and isinstance(args[1], Register)
            inputs.append(args[1])
            if args[1] == Register("lr"):
                is_return = True
            else:
                jump_target = args[1]
                is_conditional = True
                eval_fn = lambda s, a: s.set_switch_expr(a.reg(1))

        elif base in cls.instrs_store:
            assert isinstance(args[0], Register)
            inputs = [args[0]]
            is_store = True

            if isinstance(args[1], AsmAddressMode):
                inputs.append(args[1].base)

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                store = cls.instrs_store[base](a)
                if store is not None:
                    s.store_memory(store, a.reg_ref(0))

        elif base in cls.instrs_nz_flags:
            if base in ("mov", "mvn"):
                assert len(args) == 2
            elif base == "mla":
                assert len(args) == 4
            else:
                assert len(args) == 3
            assert isinstance(args[0], Register)
            outputs = [args[0]]
            inputs = get_inputs(1)
            if set_flags:
                outputs.extend([Register("n"), Register("z")])
                clobbers = [Register("hi"), Register("ge"), Register("gt")]
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                val = cls.instrs_nz_flags[base](a)
                val = s.set_reg(a.reg_ref(0), val)
                if set_flags:
                    if base in ("mov", "mul", "mla"):
                        # Guess that bit 31 represents the sign of a 32-bit integer.
                        # Use a manual cast so that the type of val is not modified
                        # until the resulting bit is .use()'d.
                        sval = as_type(val, Type.s32(), silent=True, unify=False)
                        s.set_reg(Register("n"), BinaryOp.scmp(sval, "<", Literal(0)))
                    else:
                        # Guess that it's a bit check.
                        top_bit = BinaryOp.int(val, "&", Literal(1 << 31))
                        s.set_reg(
                            Register("n"), BinaryOp.icmp(top_bit, "!=", Literal(0))
                        )
                    s.set_reg(Register("z"), BinaryOp.icmp(val, "==", Literal(0)))

        elif base in cls.instrs_mul_full:
            assert len(args) == 4
            assert isinstance(args[0], Register)
            assert isinstance(args[1], Register)
            outputs = [args[0], args[1]]
            inputs = get_inputs(2)
            if set_flags:
                outputs.extend([Register("n"), Register("z")])
                clobbers = [Register("hi"), Register("ge"), Register("gt")]
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                hi, lo = cls.instrs_mul_full[base](a)
                lo = s.set_reg(a.reg_ref(0), lo)
                hi = s.set_reg(a.reg_ref(1), hi)
                if set_flags:
                    shi = as_type(hi, Type.s32(), silent=True, unify=False)
                    s.set_reg(Register("n"), BinaryOp.scmp(shi, "<", Literal(0)))
                    s.set_reg(
                        Register("z"),
                        BinaryOp(
                            BinaryOp.icmp(lo, "==", Literal(0)),
                            "&&",
                            BinaryOp.icmp(hi, "==", Literal(0)),
                            type=Type.boolean(),
                        ),
                    )

        elif base in ("smlal", "umlal"):
            assert len(args) == 4
            assert isinstance(args[0], Register)
            assert isinstance(args[1], Register)
            outputs = [args[0], args[1], Register(base)]
            inputs = get_inputs(0)
            if set_flags:
                outputs.extend([Register("n"), Register("z")])
                clobbers = [Register("hi"), Register("ge"), Register("gt")]
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                signed = base == "smlal"
                tp = Type.sintish() if signed else Type.uintish()
                tp64 = Type.s64() if signed else Type.u64()
                lhs = as_type(a.reg(2), tp, silent=True)
                rhs = as_type(a.reg(3), tp, silent=True)
                orig = BinaryOp.int(
                    a.reg(0), "+", BinaryOp.int(a.reg(1), "<<", Literal(32))
                )
                mul = BinaryOp.int(lhs, "*", rhs)
                mlal = as_type(BinaryOp.int(orig, "+", mul), tp64, silent=True)
                mlal = s.set_reg(Register(base), mlal)
                upper = BinaryOp.int(mlal, ">>", Literal(32))
                lo = as_type(mlal, Type.int_of_size(32), silent=False)
                hi = as_type(upper, Type.int_of_size(32), silent=False)
                s.set_reg(a.reg_ref(0), lo)
                s.set_reg(a.reg_ref(1), hi)
                if set_flags:
                    as_s64 = as_type(mlal, Type.s64(), silent=True, unify=False)
                    s.set_reg(Register("n"), BinaryOp.scmp(as_s64, "<", Literal(0)))
                    s.set_reg(Register("z"), BinaryOp.icmp(mlal, "==", Literal(0)))

        elif base in ("tst", "teq"):
            assert len(args) == 2
            inputs = get_inputs(0)
            outputs = [Register("n"), Register("z")]
            clobbers = [Register("hi"), Register("ge"), Register("gt")]
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                lhs = a.reg(0)
                rhs = a.reg_or_imm(1)
                if base == "tst":
                    val = BinaryOp.int(lhs, "&", rhs)
                    s.set_reg(Register("z"), BinaryOp.icmp(val, "==", Literal(0)))
                else:
                    val = BinaryOp.int(lhs, "^", rhs)
                    s.set_reg(Register("z"), BinaryOp.icmp(lhs, "==", rhs))
                top_bit = BinaryOp.int(val, "&", Literal(1 << 31))
                s.set_reg(Register("n"), BinaryOp.icmp(top_bit, "!=", Literal(0)))

        elif mnemonic == "setcarryi.fictive":
            assert len(args) == 1 and isinstance(args[0], AsmLiteral)
            imm = args[0].value
            outputs = [Register("c"), Register("hi")]
            inputs = [Register("z")] if imm else []
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(Register("c"), Literal(imm))
                if imm == 0:
                    s.set_reg(Register("hi"), Literal(0))
                else:
                    z = condition_from_expr(a.regs[Register("z")])
                    s.set_reg(Register("hi"), z.negated())

        elif mnemonic == "setcarry.fictive":
            assert len(args) == 1 and isinstance(args[0], Register)
            outputs = [Register("c"), Register("hi")]
            inputs = [Register("z")]
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                c = s.set_reg(Register("c"), CarryBit(a.reg(0)))
                z = condition_from_expr(a.regs[Register("z")])
                hi = BinaryOp(c, "&&", z.negated(), type=Type.boolean())
                s.set_reg(Register("hi"), hi)

        elif base in cls.instrs_add:
            assert len(args) == 3 and isinstance(args[0], Register)
            outputs = [args[0]]
            if set_flags:
                outputs += cls.flag_regs
            inputs = get_inputs(1)
            if base == "adc":
                inputs.append(Register("c"))
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                val = cls.instrs_add[base](a)
                val = s.set_reg(a.reg_ref(0), val)
                if set_flags:
                    set_arm_flags_from_add(s, val)

        elif base in cls.instrs_sub:
            assert len(args) == 3 and isinstance(args[0], Register)
            outputs = [args[0]]
            if set_flags:
                outputs += cls.flag_regs
            inputs = get_inputs(1)
            if base in ("sbc", "rsc"):
                inputs.append(Register("c"))
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                val = cls.instrs_sub[base](a)
                if isinstance(val, BinaryOp):
                    val = fold_divmod(val)
                val = fold_mul_chains(val)
                val = s.set_reg(a.reg_ref(0), val)
                if set_flags:
                    s.set_reg(
                        Register("z"),
                        BinaryOp.icmp(val, "==", Literal(0, type=val.type)),
                    )
                    sval = as_type(val, Type.s32(), silent=True, unify=False)
                    s.set_reg(Register("n"), BinaryOp.scmp(sval, "<", Literal(0)))
                    v = fn_op("M2C_OVERFLOW", [sval], Type.boolean())
                    s.set_reg(Register("v"), v)
                    # Remaining flag bits are based on the full mathematical result
                    # of unsigned/signed subtractions. We don't have a good way to
                    # write that; let's cheat and treat a cast of the result to s64
                    # as the entire subtraction having been performed as s64, and
                    # hope it gets the picture across.
                    #
                    # We could special-case subs/rsbs and implement them the same way
                    # as cmp, but it might just make things less legible?
                    uval = as_type(val, Type.u32(), silent=True, unify=False)
                    sval = as_type(val, Type.s32(), silent=True, unify=False)
                    s64u = as_type(uval, Type.s64(), silent=False, unify=False)
                    s64s = as_type(sval, Type.s64(), silent=False, unify=False)
                    s.set_reg(Register("c"), BinaryOp.scmp(s64u, ">=", Literal(0)))
                    s.set_reg(Register("hi"), BinaryOp.scmp(s64u, ">", Literal(0)))
                    s.set_reg(Register("ge"), BinaryOp.scmp(s64s, ">=", Literal(0)))
                    s.set_reg(Register("gt"), BinaryOp.scmp(s64s, ">", Literal(0)))

        elif base == "cmp":
            assert len(args) == 2 and isinstance(args[0], Register)
            outputs = list(cls.flag_regs)
            inputs = get_inputs(0)
            is_effectful = False
            eval_fn = lambda s, a: eval_arm_cmp(s, a.reg(0), a.reg_or_imm(1))

        elif base == "cmn":
            assert len(args) == 2 and isinstance(args[0], Register)
            outputs = list(cls.flag_regs)
            inputs = get_inputs(0)
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                lhs = a.reg(0)
                rhs = a.reg_or_imm(1)
                if isinstance(rhs, Literal) and (rhs.value & 0xFFFFFFFF) != 0x80000000:
                    eval_arm_cmp(s, lhs, Literal(-rhs.value))
                else:
                    set_arm_flags_from_add(s, handle_add_real(lhs, rhs, a))

        elif base in cls.instrs_float_comp:
            assert (
                len(args) == 2
                and isinstance(args[0], Register)
                and isinstance(args[1], Register)
            )
            if ".d." in mnemonic:
                inputs = [
                    args[0],
                    other_f64_reg(args[0]),
                    args[1],
                    other_f64_reg(args[1]),
                ]
            else:
                inputs = [args[0], args[1]]

            cmp_cc, handler = cls.instrs_float_comp[base]
            cmp_cc, cc_negated = factor_cond(cmp_cc)
            cc_reg = CC_REGS[cmp_cc]

            clobbers = [r for r in cls.flag_regs if r != cc_reg]
            outputs = [Register("r0"), cc_reg]
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                cond = handler(a)
                s.set_reg_real(Register("r0"), cond, function_return=True)
                s.set_reg(cc_reg, cond.negated() if cc_negated else cond)

        elif base in cls.instrs_ignore:
            is_effectful = False
        else:
            # If the mnemonic is unsupported, guess if it is destination-first
            if args and isinstance(args[0], Register):
                inputs = [r for r in args[1:] if isinstance(r, Register)]
                outputs = [args[0]]
                maybe_dest_first = True
            else:
                maybe_dest_first = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                error = ErrorExpr(f"unknown instruction: {instr_str}")
                if maybe_dest_first:
                    s.set_reg_real(a.reg_ref(0), error, emit_exactly_once=True)
                else:
                    s.write_statement(ExprStmt(error))

        if cc is not None:
            inputs.append(CC_REGS[cc])

        if ShiftedRegPattern.sets_flags_based_on_barrel_shifter(base, set_flags, args):
            outputs += [Register("c"), Register("hi")]

        return Instruction(
            mnemonic=mnemonic,
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=clobbers,
            outputs=outputs,
            jump_target=jump_target,
            function_target=function_target,
            is_conditional=is_conditional,
            is_return=is_return,
            is_store=is_store,
            is_load=is_load,
            is_effectful=is_effectful,
            eval_fn=eval_fn,
        )

    asm_patterns = [
        AddPcPcPattern(),
        ShortJumpTablePattern(),
        ConditionalInstrPattern(),
        NegatedRegAddrModePattern(),
        ShiftedRegAddrModePattern(),
        ShiftedRegPattern(),
        AddrModeWritebackPattern(),
        RegRegAddrModePattern(),
        PopAndReturnPattern(),
        TailCallPattern(),
        BlBranchPattern(),
        MagicFuncPattern(),
    ]

    instrs_ignore: Set[str] = {
        "push",
        "nop",
    }

    instrs_no_flags: InstrMap = {
        # Bit-fiddling
        "clz": lambda a: fn_op("CLZ", [a.reg(1)], Type.intish()),
        "rbit": lambda a: fn_op("REVERSE_BITS", [a.reg(1)], Type.intish()),
        "rev": lambda a: fn_op("BSWAP32", [a.reg(1)], Type.intish()),
        "rev16": lambda a: fn_op("BSWAP16X2", [a.reg(1)], Type.intish()),
        "revsh": lambda a: fn_op("BSWAP16", [a.reg(1)], Type.s16()),
        # Shifts (flag-setting forms have been normalized into first movs with
        # shifted register, then shift + movs)
        "lsl": lambda a: handle_sll(a, arm=True),
        "asr": lambda a: handle_shift_right(a, arm=True, signed=True),
        "lsr": lambda a: handle_shift_right(a, arm=True, signed=False),
        "ror": lambda a: fn_op(
            "ROTATE_RIGHT", [a.reg(1), a.reg_or_imm(2)], Type.intish()
        ),
        "rrx": lambda a: fn_op(
            "ARM_RRX", [a.reg(1), a.regs[Register("c")]], Type.intish()
        ),
        # Immediate movs
        "adr": lambda a: handle_la(a),
        "movimm.fictive": lambda a: a.full_imm(1),
        # Loading instructions
        "ldr": lambda a: handle_load(a, type=Type.reg32(likely_float=False)),
        "ldrb": lambda a: handle_load(a, type=Type.u8()),
        "ldrsb": lambda a: handle_load(a, type=Type.s8()),
        "ldrh": lambda a: handle_load(a, type=Type.u16()),
        "ldrsh": lambda a: handle_load(a, type=Type.s16()),
        # Arithmetic
        "smulbb": lambda a: BinaryOp.int(a.reg(1), "*", a.reg(2)),
        "smlabb": lambda a: handle_add_real(
            a.reg(3), BinaryOp.int(a.reg(1), "*", a.reg(2)), a
        ),
        "sdiv": lambda a: BinaryOp.sint(a.reg(1), "/", a.reg(2)),
        "udiv": lambda a: BinaryOp.uint(a.reg(1), "/", a.reg(2)),
        "smod.fictive": lambda a: BinaryOp.sint(a.reg(1), "%", a.reg(2)),
        "umod.fictive": lambda a: BinaryOp.uint(a.reg(1), "%", a.reg(2)),
        # Floating point arithmetic
        "add.s.fictive": lambda a: handle_add_float(a),
        "sub.s.fictive": lambda a: BinaryOp.f32(a.reg(1), "-", a.reg(2)),
        "div.s.fictive": lambda a: BinaryOp.f32(a.reg(1), "/", a.reg(2)),
        "mul.s.fictive": lambda a: BinaryOp.f32(a.reg(1), "*", a.reg(2)),
        "sqrt.s.fictive": lambda a: fn_op("sqrtf", [as_f32(a.reg(1))], Type.f32()),
        "round.s.fictive": lambda a: fn_op("roundf", [as_f32(a.reg(1))], Type.f32()),
        "add.d.fictive": lambda a: handle_add_double(a),
        "sub.d.fictive": lambda a: BinaryOp.f64(a.dreg(1), "-", a.dreg(2)),
        "div.d.fictive": lambda a: BinaryOp.f64(a.dreg(1), "/", a.dreg(2)),
        "mul.d.fictive": lambda a: BinaryOp.f64(a.dreg(1), "*", a.dreg(2)),
        "sqrt.d.fictive": lambda a: fn_op("sqrt", [as_f64(a.dreg(1))], Type.f64()),
        "round.d.fictive": lambda a: fn_op("round", [as_f64(a.dreg(1))], Type.f64()),
        # Floating point conversions
        "cvt.d.s.fictive": lambda a: handle_convert(a.reg(1), Type.f64(), Type.f32()),
        "cvt.d.w.fictive": lambda a: handle_convert(
            a.reg(1), Type.f64(), Type.intish()
        ),
        "cvt.d.u.fictive": lambda a: handle_convert(a.reg(1), Type.f64(), Type.u32()),
        "cvt.s.d.fictive": lambda a: handle_convert(a.dreg(1), Type.f32(), Type.f64()),
        "cvt.s.w.fictive": lambda a: handle_convert(
            a.reg(1), Type.f32(), Type.intish()
        ),
        "cvt.s.u.fictive": lambda a: handle_convert(a.reg(1), Type.f32(), Type.u32()),
        "cvt.w.d.fictive": lambda a: handle_convert(a.dreg(1), Type.s32(), Type.f64()),
        "cvt.w.s.fictive": lambda a: handle_convert(a.reg(1), Type.s32(), Type.f32()),
        "cvt.u.d.fictive": lambda a: handle_convert(a.dreg(1), Type.u32(), Type.f64()),
        "cvt.u.s.fictive": lambda a: handle_convert(a.reg(1), Type.u32(), Type.f32()),
    }

    instrs_float_comp: Mapping[str, Tuple[Cc, Callable[[InstrArgs], Condition]]] = {
        "c.eq.s.fictive": (Cc.EQ, lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1))),
        "c.lt.s.fictive": (Cc.CC, lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1))),
        "c.ge.s.fictive": (Cc.CS, lambda a: BinaryOp.fcmp(a.reg(0), ">=", a.reg(1))),
        "c.le.s.fictive": (Cc.LS, lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1))),
        "c.gt.s.fictive": (Cc.HI, lambda a: BinaryOp.fcmp(a.reg(0), ">", a.reg(1))),
        "c.neq.s.fictive": (
            Cc.NE,
            lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)).negated(),
        ),
        "c.eq.d.fictive": (Cc.EQ, lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1))),
        "c.lt.d.fictive": (Cc.CC, lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1))),
        "c.ge.d.fictive": (Cc.CS, lambda a: BinaryOp.dcmp(a.dreg(0), ">=", a.dreg(1))),
        "c.le.d.fictive": (Cc.LS, lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1))),
        "c.gt.d.fictive": (Cc.HI, lambda a: BinaryOp.dcmp(a.dreg(0), ">", a.dreg(1))),
        "c.neq.d.fictive": (
            Cc.NE,
            lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)).negated(),
        ),
    }

    instrs_divmod: Dict[str, Callable[[InstrArgs], Tuple[Expression, Expression]]] = {
        "sdivmod.fictive": lambda a: (
            BinaryOp.sint(a.reg(0), "/", a.reg(1)),
            BinaryOp.sint(a.reg(0), "%", a.reg(1)),
        ),
        "udivmod.fictive": lambda a: (
            BinaryOp.uint(a.reg(0), "/", a.reg(1)),
            BinaryOp.uint(a.reg(0), "%", a.reg(1)),
        ),
    }

    instrs_nz_flags: InstrMap = {
        "mov": lambda a: handle_arm_mov(a),
        "mul": lambda a: BinaryOp.int(a.reg(1), "*", a.reg(2)),
        "mla": lambda a: BinaryOp.int(
            BinaryOp.int(a.reg(1), "*", a.reg(2)), "+", a.reg(3)
        ),
        "mvn": lambda a: handle_bitinv(a.reg_or_imm(1)),
        "and": lambda a: replace_bitand(BinaryOp.int(a.reg(1), "&", a.reg_or_imm(2))),
        "orr": lambda a: handle_or(a.reg(1), a.reg_or_imm(2), is_arm=True),
        "eor": lambda a: BinaryOp.int(a.reg(1), "^", a.reg_or_imm(2)),
        "bic": lambda a: BinaryOp.int(a.reg(1), "&", UnaryOp.int("~", a.reg_or_imm(2))),
        "orn": lambda a: BinaryOp.int(a.reg(1), "|", UnaryOp.int("~", a.reg_or_imm(2))),
    }

    instrs_mul_full: Dict[str, Callable[[InstrArgs], Tuple[Expression, Expression]]] = {
        "smull": lambda a: (
            fold_divmod(BinaryOp.int(a.reg(2), "MULT_HI", a.reg(3))),
            BinaryOp.int(a.reg(2), "*", a.reg(3)),
        ),
        "umull": lambda a: (
            fold_divmod(BinaryOp.int(a.reg(2), "MULTU_HI", a.reg(3))),
            BinaryOp.int(a.reg(2), "*", a.reg(3)),
        ),
    }

    instrs_store: StoreInstrMap = {
        "str": lambda a: make_store(a, type=Type.reg32(likely_float=False)),
        "strb": lambda a: make_store(a, type=Type.int_of_size(8)),
        "strh": lambda a: make_store(a, type=Type.int_of_size(16)),
    }

    instrs_add: InstrMap = {
        "add": lambda a: handle_add_arm(a),
        "adc": lambda a: handle_add_real(handle_add_arm(a), a.regs[Register("c")], a),
    }

    instrs_sub: InstrMap = {
        "sub": lambda a: handle_sub_arm(a),
        "rsb": lambda a: handle_sub(a.reg_or_imm(2), a.reg(1)),
        "sbc": lambda a: handle_add_real(
            handle_sub(a.reg(1), a.reg_or_imm(2)),
            condition_from_expr(a.regs[Register("c")]).negated(),
            a,
        ),
        "rsc": lambda a: handle_add_real(
            handle_sub(a.reg_or_imm(2), a.reg(1)),
            condition_from_expr(a.regs[Register("c")]).negated(),
            a,
        ),
    }

    def default_function_abi_candidate_slots(self) -> List[AbiArgSlot]:
        return [
            AbiArgSlot(ArgLoc(None, 0, Register("r0")), Type.any_reg()),
            AbiArgSlot(ArgLoc(None, 1, Register("r1")), Type.any_reg()),
            AbiArgSlot(ArgLoc(None, 2, Register("r2")), Type.any_reg()),
            AbiArgSlot(ArgLoc(None, 3, Register("r3")), Type.any_reg()),
        ]

    def is_likely_partial_offset(self, addend: int) -> bool:
        return addend < 0x1000000 and addend % 0x100 == 0

    def arg_name(self, loc: ArgLoc) -> str:
        if loc.offset is not None:
            return f"arg{format_hex(loc.offset // 4 + 4)}"
        assert loc.reg is not None
        return f"arg{loc.reg.register_name[1:]}"

    def function_abi(
        self,
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> Abi:
        """Compute stack positions/registers used by a function according to the agbcc ABI,
        based on C type information. Additionally computes a list of registers that might
        contain arguments, if the function is a varargs function. (Additional varargs
        arguments may be passed on the stack; we could compute the offset at which that
        would start but right now don't care -- we just slurp up everything.)"""

        known_slots: List[AbiArgSlot] = []
        candidate_slots: List[AbiArgSlot] = []
        if fn_sig.params_known:
            offset = 0
            if (
                fn_sig.return_type.is_struct()
                and fn_sig.return_type.get_parameter_size_align_bytes()[0] > 4
            ):
                # The ABI for struct returns is to pass a pointer to where it should be written
                # as the first argument.
                known_slots.append(
                    AbiArgSlot(
                        ArgLoc(None, 0, Register("r0")),
                        Type.ptr(fn_sig.return_type),
                        name="__return__",
                        comment="return",
                    )
                )
                offset = 4

            for param in fn_sig.params:
                # Array parameters decay into pointers
                param_type = param.type.decay()
                size, align = param_type.get_parameter_size_align_bytes()
                size = (size + 3) & ~3
                offset = (offset + align - 1) & -align
                name = param.name
                for i in range(offset // 4, (offset + size) // 4):
                    unk_offset = 4 * i - offset
                    reg2: Optional[Register] = None
                    stack_loc: Optional[int] = None
                    if i < 4:
                        reg2 = Register(f"r{i}")
                    else:
                        stack_loc = (i - 4) * 4
                    if size > 4:
                        name2 = f"{name}_unk{unk_offset:X}" if name else None
                        sub_type = Type.any()
                        comment: Optional[str] = f"{param_type}+{unk_offset:#x}"
                    else:
                        assert unk_offset == 0
                        name2 = name
                        sub_type = param_type
                        comment = None
                    known_slots.append(
                        AbiArgSlot(
                            ArgLoc(stack_loc, i, reg2),
                            sub_type,
                            name=name2,
                            comment=comment,
                        )
                    )
                offset += size

            if fn_sig.is_variadic:
                for i in range(offset // 4, 4):
                    candidate_slots.append(
                        AbiArgSlot(ArgLoc(None, i, Register(f"r{i}")), Type.any_reg())
                    )

        else:
            candidate_slots = self.default_function_abi_candidate_slots()

        valid_extra_regs: Set[Register] = {
            slot.loc.reg for slot in known_slots if slot.loc.reg is not None
        }
        possible_slots: List[AbiArgSlot] = []

        # If register rX is likely, all previous ones are as well.
        lasti = -1
        for i in range(4):
            if Register(f"r{i}") not in likely_regs:
                break
            lasti = i
        for i in range(lasti, 0, -1):
            if likely_regs[Register(f"r{i}")]:
                likely_regs[Register(f"r{i - 1}")] = True

        for slot in candidate_slots:
            reg = slot.loc.reg
            if reg is None or reg not in likely_regs:
                continue

            # Don't pass this register if lower numbered ones are undefined.
            require: Optional[List[str]] = None
            if slot == candidate_slots[0]:
                # For varargs, a subset of r0 .. r3 may be used. Don't check
                # earlier registers for the first member of that subset.
                pass
            elif reg == Register("r1"):
                require = ["r0"]
            elif reg == Register("r2"):
                require = ["r1"]
            elif reg == Register("r3"):
                require = ["r2"]
            if require and not any(Register(r) in valid_extra_regs for r in require):
                continue

            valid_extra_regs.add(reg)

            # Skip registers that are untouched from the initial parameter
            # list. This is sometimes wrong (can give both false positives
            # and negatives), but having a heuristic here is unavoidable
            # without access to function signatures, or when dealing with
            # varargs functions. Decompiling multiple functions at once
            # would help.
            if not likely_regs[reg]:
                continue

            possible_slots.append(slot)

        return Abi(
            arg_slots=known_slots,
            possible_slots=possible_slots,
        )

    @staticmethod
    def function_return(expr: Expression) -> Dict[Register, Expression]:
        # We may not know what this function's return registers are --
        # $r0 or ($r0,$r1) -- but we don't really care, it's fine to be
        # liberal here and put the return value in all of them.
        # (It's not perfect for u64's, but that's rare anyway.)
        return {
            Register("r0"): as_type(expr, Type.intptr(), silent=True, unify=False),
            Register("r1"): fn_op("SECOND_REG", [expr], Type.reg32(likely_float=False)),
        }
