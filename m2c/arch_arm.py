from __future__ import annotations
from dataclasses import replace
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .error import DecompFailure
from .options import Target
from .asm_file import Label
from .asm_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    BinOp,
    JumpTarget,
    Register,
    get_jump_target,
)
from .asm_pattern import (
    AsmMatch,
    AsmMatcher,
    AsmPattern,
    Replacement,
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
    CommentStmt,
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
    as_f32,
    as_f64,
    as_intish,
    as_s64,
    as_sintish,
    as_type,
    as_u32,
    as_u64,
    as_uintish,
)
from .evaluate import (
    condition_from_expr,
    error_stmt,
    fn_op,
    fold_divmod,
    fold_mul_chains,
    handle_add,
    handle_add_double,
    handle_add_float,
    handle_add_real,
    handle_addi,
    handle_bgez,
    handle_conditional_move,
    handle_convert,
    handle_la,
    handle_lw,
    handle_load,
    handle_lwl,
    handle_lwr,
    handle_or,
    handle_sltiu,
    handle_sltu,
    handle_sra,
    handle_swl,
    handle_swr,
    imm_add_32,
    load_upper,
    make_store,
    void_fn_op,
)
from .types import FunctionSignature, Type


LENGTH_THREE: Set[str] = {
    "add",
    "adc",
    "sub",
    "sbc",
    "rsb",
    "rsc",
    "mul",
    "eor",
    "orr",
    "and",
    "bic",
    "lsl",
    "asr",
    "lsr",
    "ror",
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


def parse_suffix(mnemonic: str) -> Tuple[str, Optional[Cc], bool]:
    set_flags = False
    if mnemonic.endswith("s"):
        mnemonic = mnemonic[:-1]
        set_flags = True
    cc: Optional[Cc] = None
    for suffix in [cond.value for cond in Cc] + ["hs", "lo"]:
        if mnemonic.endswith(suffix):
            if suffix == "hs":
                cc = Cc.CS
            elif suffix == "lo":
                cc = Cc.CC
            else:
                cc = Cc(suffix)
            mnemonic = mnemonic[: -len(suffix)]
            break
    if mnemonic.endswith("s") and not set_flags:
        mnemonic = mnemonic[:-1]
        set_flags = True
    return mnemonic, cc, set_flags


class ConditionalInstrPattern(AsmPattern):
    """
    Replace conditionally executed instructions by branches.
    """

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        matched_cc: Optional[Cc] = None
        i = 0
        if_instrs: List[AsmInstruction] = []
        else_instrs: List[AsmInstruction] = []
        while matcher.index + i < len(matcher.input):
            instr = matcher.input[matcher.index]
            if not isinstance(instr, Instruction):
                break
            if i != 0 and instr.mnemonic == "nop":
                i += 1
                continue
            base, cc, set_flags = parse_suffix(instr.mnemonic)
            if cc is None or base == "b":
                break
            new_instr = AsmInstruction(base + ("s" if set_flags else ""), instr.args)
            if matched_cc is None:
                matched_cc = cc
            if matched_cc == cc:
                if_instrs.append(new_instr)
            elif matched_cc == negate_cond(cc):
                else_instrs.append(new_instr)
            else:
                break
            i += 1
            # TODO: come up with a better check for flag clobbers?
            if set_flags or base in ("cmp", "cmn", "tst", "teq"):
                break
        if matched_cc is None:
            return None

        b_mn = "b" + negate_cond(matched_cc).value
        label1 = f"._m2c_cc_{matcher.index}"
        label2 = f"._m2c_cc2_{matcher.index}"
        if else_instrs:
            return Replacement(
                [
                    AsmInstruction(b_mn, [AsmGlobalSymbol(label1)]),
                    *if_instrs,
                    AsmInstruction("b", [AsmGlobalSymbol(label2)]),
                    Label([label1]),
                    *else_instrs,
                    Label([label2]),
                ],
                i,
            )
        else:
            return Replacement(
                [
                    AsmInstruction(b_mn, [AsmGlobalSymbol(label1)]),
                    *if_instrs,
                    Label([label1]),
                ],
                i,
            )


class PopAndReturnPattern(SimpleAsmPattern):
    pattern = make_pattern(
        "pop {x}",
        "bx $x",
    )

    def replace(self, m: AsmMatch) -> Replacement:
        return Replacement([AsmInstruction("bx", [Register("lr")])], len(m.body))


class ArmArch(Arch):
    arch = Target.ArchEnum.ARM

    re_comment = r"^[ \t]*#.*|[@;].*"
    supports_dollar_regs = False

    stack_pointer_reg = Register("sp")
    frame_pointer_reg = Register("r11")
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
        "fp": Register("r11"),
        "ip": Register("r12"),
        "sb": Register("r9"),
        "tr": Register("r9"),
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
    def normalize_instruction(cls, instr: AsmInstruction) -> AsmInstruction:
        if instr.mnemonic.endswith(".n") or instr.mnemonic.endswith(".w"):
            instr = replace(instr, mnemonic=instr.mnemonic[:-2])
        base, cc, set_flags = parse_suffix(instr.mnemonic)
        suffix = (cc.value if cc else "") + ("s" if set_flags else "")
        args = instr.args
        if cc == Cc.AL:
            return cls.normalize_instruction(
                AsmInstruction(base + ("s" if set_flags else ""), args)
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
            if base == "rrx":
                return AsmInstruction(
                    "mov" + suffix, [args[0], BinOp(base, args[1], AsmLiteral(1))]
                )
            if base in LENGTH_THREE:
                return cls.normalize_instruction(
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
        jump_target: Optional[Union[JumpTarget, Register]] = None
        function_target: Optional[Argument] = None
        is_conditional = False
        is_return = False
        is_store = False
        eval_fn: Optional[Callable[[NodeState, InstrArgs], object]] = None

        instr_str = str(AsmInstruction(mnemonic, args))

        base, cc, set_flags = parse_suffix(mnemonic)

        if base == "b":
            # Conditional or unconditional branch
            assert len(args) == 1
            jump_target = get_jump_target(args[0])

            if cc is not None:
                cc, negated = factor_cond(cc)
                is_conditional = True
                inputs = [
                    {
                        Cc.EQ: Register("z"),
                        Cc.CS: Register("c"),
                        Cc.MI: Register("n"),
                        Cc.VS: Register("v"),
                        Cc.HI: Register("hi"),
                        Cc.GE: Register("ge"),
                        Cc.GT: Register("gt"),
                    }[cc]
                ]

                def eval_fn(s: NodeState, a: InstrArgs) -> None:
                    cond: Expression
                    if cc == Cc.EQ:
                        cond = a.regs[Register("z")]
                    elif cc == Cc.CS:
                        cond = a.regs[Register("c")]
                    elif cc == Cc.MI:
                        cond = a.regs[Register("n")]
                    elif cc == Cc.VS:
                        cond = a.regs[Register("v")]
                    elif cc == Cc.HI:
                        cond = a.regs[Register("hi")]
                    elif cc == Cc.GE:
                        cond = a.regs[Register("ge")]
                    elif cc == Cc.GT:
                        cond = a.regs[Register("gt")]
                    else:
                        assert False
                    cond = condition_from_expr(cond)
                    if negated:
                        cond = cond.negated()
                    s.set_branch_condition(cond)

        elif mnemonic in ("cbz", "cbnz"):
            # Thumb conditional branch
            assert len(args) == 2
            assert isinstance(args[0], Register)
            inputs = [args[0]]
            jump_target = get_jump_target(args[1])
            is_conditional = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                op = "==" if mnemonic == "cbz" else "!="
                s.set_branch_condition(BinaryOp.icmp(a.reg(0), op, Literal(0)))

        elif mnemonic == "bx" and args[0] == Register("lr"):
            # Return
            assert len(args) == 1
            inputs = [Register("lr")]
            is_return = True
        elif mnemonic in cls.instrs_ignore:
            pass
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
            eval_fn=eval_fn,
        )

    asm_patterns = [
        ConditionalInstrPattern(),
        PopAndReturnPattern(),
    ]

    instrs_ignore: Set[str] = {
        "nop",
    }

    def default_function_abi_candidate_slots(self) -> List[AbiArgSlot]:
        # TODO: these stack locations are wrong, registers don't have pre-defined
        # home space outside of MIPS.
        return [
            AbiArgSlot(0, Register("a0"), Type.any_reg()),
            AbiArgSlot(4, Register("a1"), Type.any_reg()),
            AbiArgSlot(8, Register("a2"), Type.any_reg()),
            AbiArgSlot(12, Register("a3"), Type.any_reg()),
        ]

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
        for slot in candidate_slots:
            if slot.reg is None or slot.reg not in likely_regs:
                continue

            # Don't pass this register if lower numbered ones are undefined.
            require: Optional[List[str]] = None
            if slot == candidate_slots[0]:
                # For varargs, a subset of a0 .. a3 may be used. Don't check
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
            # TODO: don't do this in the middle of the argument list.
            if not likely_regs[slot.reg]:
                continue

            possible_slots.append(slot)

        return Abi(
            arg_slots=known_slots,
            possible_slots=possible_slots,
        )

    @staticmethod
    def function_return(expr: Expression) -> Dict[Register, Expression]:
        # We may not know what this function's return registers are --
        # $v0 or ($v0,$v1) -- but we don't really care, it's fine to be
        # liberal here and put the return value in all of them.
        # (It's not perfect for u64's, but that's rare anyway.)
        return {
            Register("v0"): Cast(
                expr, reinterpret=True, silent=True, type=Type.intptr()
            ),
            Register("v1"): as_u32(
                Cast(expr, reinterpret=True, silent=False, type=Type.u64())
            ),
        }
