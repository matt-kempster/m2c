from __future__ import annotations
from dataclasses import replace
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
    Arch,
    BinaryOp,
    Cast,
    ErrorExpr,
    ExprStmt,
    Expression,
    InstrArgs,
    InstrMap,
    Literal,
    NodeState,
    StmtInstrMap,
    StoreInstrMap,
    UnaryOp,
    as_intish,
    as_type,
    as_uintish,
)
from .evaluate import (
    condition_from_expr,
    error_stmt,
    eval_arm_cmp,
    fn_op,
    fold_divmod,
    fold_mul_chains,
    handle_add,
    handle_add_arm,
    handle_add_real,
    handle_addi,
    handle_arm_mov,
    handle_bitinv,
    handle_load,
    handle_or,
    handle_shift_right,
    handle_sll,
    handle_sub,
    handle_sub_arm,
    make_store,
    replace_bitand,
    set_arm_flags_from_add,
    void_fn_op,
)
from .types import FunctionSignature, Type


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
    # Deal with false positives from naively stripping cc/s
    if mnemonic in ("teq", "mls", "smmls"):
        return mnemonic, None, "", ""
    if mnemonic.endswith("s"):
        base = mnemonic[:-1]
        if base in (
            "mul",
            "lsl",
            "umull",
            "umlal",
            "smull",
            "smlal",
            "mov",
            "bic",
            "adc",
            "sbc",
            "rsc",
        ):
            return base, None, "s", ""

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

    direction = ""
    if any(mnemonic.endswith(suffix) for suffix in ("ia", "ib", "da", "db")):
        direction = mnemonic[-2:]
        mnemonic = mnemonic[:-2]

    memsize = ""
    if mnemonic.startswith("str") or mnemonic.startswith("ldr"):
        for suffix in ("b", "h", "d"):
            if mnemonic.endswith(suffix):
                mnemonic = mnemonic[:-1]
                memsize = suffix
                break
    if memsize in ("b", "h") and mnemonic.endswith("s") and not strip_cc(mnemonic)[1]:
        mnemonic = mnemonic[:-1]
        memsize = "s" + memsize

    mnemonic, cc = strip_cc(mnemonic)
    set_flags = ""
    if mnemonic.endswith("s"):
        set_flags = "s"
        mnemonic = mnemonic[:-1]
    if cc is None:
        mnemonic, cc = strip_cc(mnemonic)
    return mnemonic + memsize, cc, set_flags, direction


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


class ConditionalInstrPattern(AsmPattern):
    """Replace conditionally executed instructions by branches."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        matched_cc: Optional[Cc] = None
        i = 0
        if_instrs: List[AsmInstruction] = []
        else_instrs: List[AsmInstruction] = []
        before_instrs: List[Instruction] = []
        after_instrs: List[Instruction] = []
        mid_inputs: Set[Location] = set()
        mid_outputs: Set[Location] = set()
        mid_impure = False
        after_inputs: Set[Location] = set()
        after_outputs: Set[Location] = set()
        after_impure = False
        moved_before_streak = 0
        moved_after_streak = 0
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
                if (
                    not set(instr.inputs + outputs) & mid_outputs
                    and not set(outputs) & mid_inputs
                    and not after_instrs
                    and not (mid_impure and not instr.is_pure)
                ):
                    before_instrs.append(instr)
                    moved_before_streak += 1
                else:
                    after_outputs |= set(instr.outputs + instr.clobbers)
                    after_inputs |= set(instr.inputs)
                    after_instrs.append(instr)
                    moved_after_streak += 1
                    if not instr.is_pure:
                        after_impure = True
                i += 1
                continue
            outputs = instr.outputs + instr.clobbers
            if set(instr.inputs + outputs) & after_outputs:
                break
            if set(outputs) & after_inputs:
                break
            if after_impure and not instr.is_pure:
                break
            if after_instrs and instr.is_jump():
                # If we have a jump, there is no way to put instructions after.
                break
            mid_outputs |= set(outputs)
            mid_inputs |= set(instr.inputs)
            if not instr.is_pure:
                mid_impure = True
            new_instr = AsmInstruction(base + set_flags + direction, instr.args)
            if matched_cc is None:
                if base == "b":
                    break
                matched_cc = cc
            if matched_cc == cc:
                if_instrs.append(new_instr)
            elif matched_cc == negate_cond(cc):
                else_instrs.append(new_instr)
            else:
                break
            i += 1
            moved_before_streak = 0
            moved_after_streak = 0
            checked_reg = CC_REGS[factor_cond(matched_cc)[0]]
            if checked_reg in outputs or instr.is_jump():
                break
        if matched_cc is None:
            return None

        i -= moved_before_streak + moved_after_streak
        del before_instrs[len(before_instrs) - moved_before_streak :]
        del after_instrs[len(after_instrs) - moved_after_streak :]

        b_mn = "b" + negate_cond(matched_cc).value
        label1 = f"._m2c_cc_{matcher.index}"
        label2 = f"._m2c_cc2_{matcher.index}"
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

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        instr = matcher.input[matcher.index]
        if not isinstance(instr, Instruction) or not instr.args:
            return None
        base, cc, set_flags, direction = parse_suffix(instr.mnemonic)

        arg = instr.args[-1]
        literal_carry: Optional[int] = None
        new_instrs: List[ReplacementPart]
        if isinstance(arg, AsmLiteral):
            # Carry = bit 31 if the constant is greater than 255 and can be
            # produced by shifting an 8-bit value
            value = arg.value & 0xFFFFFFFF
            if value < 0x100 or value >= (value & -value) * 0x100:
                if base in ("mov", "mvn"):
                    value ^= 0xFFFFFFFF
                if value < 0x100 or value >= (value & -value) * 0x100:
                    return None
            new_instrs = [instr]
            literal_carry = value >> 31
        elif isinstance(arg, BinOp) and arg.op in ARM_BARREL_SHIFTER_OPS:
            temp = Register.fictive(arg.op)
            shift_ins = AsmInstruction(arg.op, [temp, arg.lhs, arg.rhs])
            if arg.op == "rrx":
                shift_ins.args.pop()
            new_instrs = [
                shift_ins,
                AsmInstruction(instr.mnemonic, instr.args[:-1] + [temp]),
            ]
        else:
            return None

        if base in ("tst", "teq") or (
            base in ("mov", "mvn", "and", "eor", "orr", "orn", "bic") and set_flags
        ):
            # The instruction sets carry based on the barrel shifter
            if literal_carry is not None:
                new_instrs.append(
                    AsmInstruction("setcarryi.fictive", [AsmLiteral(literal_carry)])
                )
            else:
                new_instrs.append(AsmInstruction("setcarry.fictive", [temp]))

        return Replacement(new_instrs, 1)


class PopAndReturnPattern(SimpleAsmPattern):
    pattern = make_pattern(
        "pop {x}",
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


class ArmArch(Arch):
    arch = Target.ArchEnum.ARM

    re_comment = r"^[ \t]*#.*|[@;].*"
    supports_dollar_regs = False

    home_space_size = 0

    stack_pointer_reg = Register("sp")
    frame_pointer_regs = [Register("r7"), Register("r11")]  # for Thumb/ARM respectively
    return_address_reg = Register("lr")

    base_return_regs = [(Register("r0"), False)]
    all_return_regs = [Register("r0")]
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
        if instr.mnemonic.endswith(".n") or instr.mnemonic.endswith(".w"):
            instr = replace(instr, mnemonic=instr.mnemonic[:-2])
        base, cc, set_flags, direction = parse_suffix(instr.mnemonic)
        cc_str = cc.value if cc else ""
        suffix = cc_str + set_flags + direction
        args = instr.args
        if cc == Cc.AL:
            return cls.normalize_instruction_real(
                AsmInstruction(base + set_flags + direction, args)
            )
        if len(args) == 3:
            if base in ("add", "lsl") and args[2] == AsmLiteral(0):
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
        is_store = False
        is_pure = False
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
                # TODO
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
            assert isinstance(args[0], Register)
            outputs = [args[0]]
            inputs = get_inputs(1)
            is_pure = not base.startswith("ldr")

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(a.reg_ref(0), cls.instrs_no_flags[base](a))

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
            is_pure = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                val = cls.instrs_nz_flags[base](a)
                val = s.set_reg(a.reg_ref(0), val)
                if set_flags:
                    if base in ("mov", "mul", "mla"):
                        # Guess that bit 31 represents the sign of a 32-bit integer.
                        # Use a manual cast so that the type of val is not modified
                        # until the resulting bit is .use()'d.
                        sval = Cast(val, reinterpret=True, silent=True, type=Type.s32())
                        s.set_reg(Register("n"), BinaryOp.scmp(sval, "<", Literal(0)))
                    else:
                        # Guess that it's a bit check.
                        top_bit = BinaryOp.int(val, "&", Literal(1 << 31))
                        s.set_reg(
                            Register("n"), BinaryOp.icmp(top_bit, "!=", Literal(0))
                        )
                    s.set_reg(Register("z"), BinaryOp.icmp(val, "==", Literal(0)))

        elif base in ("tst", "teq"):
            assert len(args) == 2
            inputs = get_inputs(0)
            outputs = [Register("n"), Register("z")]
            clobbers = [Register("hi"), Register("ge"), Register("gt")]
            is_pure = True

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
            is_pure = True

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
            is_pure = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                c = s.set_reg(
                    Register("c"), fn_op("M2C_CARRY", [a.reg(0)], Type.bool())
                )
                z = condition_from_expr(a.regs[Register("z")])
                hi = BinaryOp(c, "&&", z.negated(), type=Type.bool())
                s.set_reg(Register("hi"), hi)

        elif base in cls.instrs_add:
            assert len(args) == 3 and isinstance(args[0], Register)
            outputs = [args[0]]
            if set_flags:
                outputs += cls.flag_regs
            inputs = get_inputs(1)
            if base == "adc":
                inputs.append(Register("c"))
            is_pure = True

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
            is_pure = True

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
                    sval = Cast(val, reinterpret=True, silent=True, type=Type.s32())
                    s.set_reg(Register("n"), BinaryOp.scmp(val, "<", Literal(0)))
                    v = fn_op("M2C_OVERFLOW", [val], Type.bool())
                    s.set_reg(Register("v"), v)
                    # Remaining flag bits are based on the full mathematical result
                    # of unsigned/signed subtractions. We don't have a good way to
                    # write that; let's cheat and treat a cast of the result to s64
                    # as the entire subtraction having been performed as s64, and
                    # hope it gets the picture across.
                    #
                    # We could special-case subs/rsbs and implement them the same way
                    # as cmp, but it might just make things less legible?
                    uval = Cast(val, reinterpret=True, silent=True, type=Type.u32())
                    sval = Cast(val, reinterpret=True, silent=True, type=Type.s32())
                    s64u = Cast(uval, reinterpret=True, silent=False, type=Type.s64())
                    s64s = Cast(sval, reinterpret=True, silent=False, type=Type.s64())
                    s.set_reg(Register("c"), BinaryOp.scmp(s64u, ">=", Literal(0)))
                    s.set_reg(Register("hi"), BinaryOp.scmp(s64u, ">", Literal(0)))
                    s.set_reg(Register("ge"), BinaryOp.scmp(s64s, ">=", Literal(0)))
                    s.set_reg(Register("gt"), BinaryOp.scmp(s64s, ">", Literal(0)))

        elif base == "cmp":
            assert len(args) == 2 and isinstance(args[0], Register)
            outputs = list(cls.flag_regs)
            inputs = get_inputs(0)
            is_pure = True
            eval_fn = lambda s, a: eval_arm_cmp(s, a.reg(0), a.reg_or_imm(1))

        elif base == "cmn":
            assert len(args) == 2 and isinstance(args[0], Register)
            outputs = list(cls.flag_regs)
            inputs = get_inputs(0)
            is_pure = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                lhs = a.reg(0)
                rhs = a.reg_or_imm(1)
                if isinstance(rhs, Literal) and (rhs.value & 0xFFFFFFFF) != 0x80000000:
                    eval_arm_cmp(s, lhs, Literal(-rhs.value))
                else:
                    set_arm_flags_from_add(s, handle_add_arm(a))

        elif base in cls.instrs_ignore:
            is_pure = True
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
            is_pure=is_pure,
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
        # Shifts (flag-setting forms have been normalized into shift + movs)
        "lsl": lambda a: handle_sll(a, arm=True),
        "asr": lambda a: handle_shift_right(a, arm=True, signed=True),
        "lsr": lambda a: handle_shift_right(a, arm=True, signed=False),
        "ror": lambda a: fn_op(
            "ROTATE_RIGHT", [a.reg(1), a.reg_or_imm(2)], Type.intish()
        ),
        "rrx": lambda a: fn_op(
            "ARM_RRX", [a.reg(1), a.regs[Register("c")]], Type.intish()
        ),
        # Loading instructions
        "ldr": lambda a: handle_load(a, type=Type.reg32(likely_float=False)),
        "ldrb": lambda a: handle_load(a, type=Type.u8()),
        "ldrsb": lambda a: handle_load(a, type=Type.s8()),
        "ldrh": lambda a: handle_load(a, type=Type.u16()),
        "ldrsh": lambda a: handle_load(a, type=Type.s16()),
    }

    instrs_nz_flags: InstrMap = {
        "mov": lambda a: handle_arm_mov(a),
        "mul": lambda a: BinaryOp.int(a.reg(1), "*", a.reg(2)),
        "mla": lambda a: BinaryOp.int(
            BinaryOp.int(a.reg(1), "*", a.reg(2)), "+", a.reg(3)
        ),
        "mvn": lambda a: handle_bitinv(a.reg_or_imm(1)),
        "and": lambda a: replace_bitand(BinaryOp.int(a.reg(1), "&", a.reg_or_imm(2))),
        "orr": lambda a: handle_or(a.reg(1), a.reg_or_imm(2)),
        "eor": lambda a: BinaryOp.int(a.reg(1), "^", a.reg_or_imm(2)),
        "bic": lambda a: BinaryOp.int(a.reg(1), "&", UnaryOp.int("~", a.reg_or_imm(2))),
        "orn": lambda a: BinaryOp.int(a.reg(1), "|", UnaryOp.int("~", a.reg_or_imm(2))),
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
        "sbc": lambda a: BinaryOp.int(
            handle_sub(a.reg(1), a.reg_or_imm(2)),
            "+",
            BinaryOp.int(Literal(1), "-", a.regs[Register("c")]),
        ),
        "rsc": lambda a: BinaryOp.int(
            handle_sub(a.reg_or_imm(2), a.reg(1)),
            "+",
            BinaryOp.int(Literal(1), "-", a.regs[Register("c")]),
        ),
    }

    def default_function_abi_candidate_slots(self) -> List[AbiArgSlot]:
        # TODO: these stack locations are wrong, registers don't have pre-defined
        # home space outside of MIPS.
        return [
            AbiArgSlot(0, Register("r0"), Type.any_reg()),
            AbiArgSlot(4, Register("r1"), Type.any_reg()),
            AbiArgSlot(8, Register("r2"), Type.any_reg()),
            AbiArgSlot(12, Register("r3"), Type.any_reg()),
        ]

    def is_likely_partial_offset(self, addend: int) -> bool:
        return addend < 0x1000000 and addend % 0x100 == 0

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
            if fn_sig.return_type.is_struct():
                # The ABI for struct returns is to pass a pointer to where it should be written
                # as the first argument.
                # TODO is this right?
                known_slots.append(
                    AbiArgSlot(
                        offset=0,
                        reg=Register("r0"),
                        name="__return__",
                        type=Type.ptr(fn_sig.return_type),
                        comment="return",
                    )
                )
                offset = 4

            for ind, param in enumerate(fn_sig.params):
                # Array parameters decay into pointers
                param_type = param.type.decay()
                size, align = param_type.get_parameter_size_align_bytes()
                size = (size + 3) & ~3
                offset = (offset + align - 1) & -align
                name = param.name
                reg2: Optional[Register]
                for i in range(offset // 4, (offset + size) // 4):
                    unk_offset = 4 * i - offset
                    reg2 = Register(f"r{i}") if i < 4 else None
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
                            offset=4 * i,
                            reg=reg2,
                            name=name2,
                            type=sub_type,
                            comment=comment,
                        )
                    )
                offset += size

            if fn_sig.is_variadic:
                for i in range(offset // 4, 4):
                    candidate_slots.append(
                        AbiArgSlot(i * 4, Register(f"r{i}"), Type.any_reg())
                    )

        else:
            candidate_slots = self.default_function_abi_candidate_slots()

        valid_extra_regs: Set[Register] = {
            slot.reg for slot in known_slots if slot.reg is not None
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
            if slot.reg is None or slot.reg not in likely_regs:
                continue

            # Don't pass this register if lower numbered ones are undefined.
            require: Optional[List[str]] = None
            if slot == candidate_slots[0]:
                # For varargs, a subset of r0 .. r3 may be used. Don't check
                # earlier registers for the first member of that subset.
                pass
            elif slot.reg == Register("r1"):
                require = ["r0"]
            elif slot.reg == Register("r2"):
                require = ["r1"]
            elif slot.reg == Register("r3"):
                require = ["r2"]
            if require and not any(Register(r) in valid_extra_regs for r in require):
                continue

            valid_extra_regs.add(slot.reg)

            # Skip registers that are untouched from the initial parameter
            # list. This is sometimes wrong (can give both false positives
            # and negatives), but having a heuristic here is unavoidable
            # without access to function signatures, or when dealing with
            # varargs functions. Decompiling multiple functions at once
            # would help.
            if not likely_regs[slot.reg]:
                continue

            possible_slots.append(slot)

        return Abi(
            arg_slots=known_slots,
            possible_slots=possible_slots,
        )

    @staticmethod
    def function_return(expr: Expression) -> Dict[Register, Expression]:
        # TODO: support r1 as a return register used for 64-bit values. We could
        # easily add something like:
        #   Register("r1"): fn_op("SECOND_REG", [expr], Type.reg32(likely_float=False)),
        # to this dict, but it causes problems with this SECOND_REG being passed as
        # argument to functions.
        return {
            Register("r0"): Cast(
                expr, reinterpret=True, silent=True, type=Type.intptr()
            ),
        }
