from typing import (
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from .error import DecompFailure
from .options import Target
from .parse_instruction import (
    Access,
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    Instruction,
    InstructionMeta,
    JumpTarget,
    Macro,
    MemoryAccess,
    Register,
    StackAccess,
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
from .translate import (
    Abi,
    AbiArgSlot,
    Arch,
    BinaryOp,
    CarryBit,
    Cast,
    CmpInstrMap,
    CommentStmt,
    ErrorExpr,
    Expression,
    ImplicitInstrMap,
    InstrMap,
    InstrSet,
    Literal,
    PpcCmpInstrMap,
    PairInstrMap,
    SecondF64Half,
    StmtInstrMap,
    StoreInstrMap,
    TernaryOp,
    UnaryOp,
    as_f32,
    as_f64,
    as_int64,
    as_intish,
    as_intptr,
    as_ptr,
    as_s32,
    as_type,
    as_u32,
    fn_op,
    fold_divmod,
    fold_mul_chains,
    handle_add,
    handle_add_double,
    handle_add_float,
    handle_addi,
    handle_addis,
    handle_convert,
    handle_load,
    handle_loadx,
    handle_or,
    handle_rlwinm,
    handle_rlwimi,
    handle_sra,
    load_upper,
    make_store,
    make_storex,
    void_fn_op,
)
from .types import FunctionSignature, Type


class FcmpoCrorPattern(SimpleAsmPattern):
    """
    For floating point, `x <= y` and `x >= y` use `cror` to OR together the `cr0_eq`
    bit with either `cr0_lt` or `cr0_gt`. Instead of implementing `cror`, we detect
    this pattern and and directly compute the two registers.
    """

    pattern = make_pattern(
        "fcmpo $cr0, $x, $y",
        "cror 2, N, 2",
    )

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        fcmpo = m.body[0]
        assert isinstance(fcmpo, Instruction)
        if m.literals["N"] == 0:
            return Replacement(
                [AsmInstruction("fcmpo.lte.fictive", fcmpo.args)], len(m.body)
            )
        elif m.literals["N"] == 1:
            return Replacement(
                [AsmInstruction("fcmpo.gte.fictive", fcmpo.args)], len(m.body)
            )
        return None


class TailCallPattern(AsmPattern):
    """
    If a function ends in `return fn(...);` then the compiler may perform tail-call
    optimization. This is emitted as `b fn` instead of using `bl fn; blr`.
    """

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        if matcher.index != len(matcher.input) - 1:
            return None
        instr = matcher.input[matcher.index]
        if (
            isinstance(instr, Instruction)
            and instr.mnemonic == "b"
            and isinstance(instr.args[0], AsmGlobalSymbol)
        ):
            return Replacement(
                [
                    AsmInstruction("bl", instr.args),
                    AsmInstruction("blr", []),
                ],
                1,
            )
        return None


class BoolCastPattern(SimpleAsmPattern):
    """Cast to bool (a 1 bit type in MWCC), which also can be emitted from `!!x`."""

    pattern = make_pattern(
        "neg $a, $x",
        "addic $r0, $a, -1",
        "subfe $r0, $r0, $a",
    )

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        body: List[ReplacementPart] = [
            AsmInstruction("boolcast.fictive", [Register("r0"), m.regs["x"]])
        ]
        if m.regs["a"] != Register("r0"):
            # Preserve `neg $a, $x` in case $a is accessed later, unless $a is $r0
            body.append(m.body[0])
        return Replacement(body, len(m.body))


class BranchCtrPattern(AsmPattern):
    """Split decrement-$ctr-and-branch instructions into a pair of instructions."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        instr = matcher.input[matcher.index]
        if isinstance(instr, Instruction) and instr.mnemonic in ("bdz", "bdnz"):
            ctr = Register("ctr")
            return Replacement(
                [
                    AsmInstruction("addi", [ctr, ctr, AsmLiteral(-1)]),
                    AsmInstruction(instr.mnemonic + ".fictive", instr.args),
                ],
                1,
            )
        return None


class PpcArch(Arch):
    arch = Target.ArchEnum.PPC

    stack_pointer_reg = Register("r1")
    frame_pointer_reg = None
    return_address_reg = Register("lr")

    base_return_regs = [Register(r) for r in ["r3", "f1"]]
    all_return_regs = [Register(r) for r in ["f1", "r3", "r4"]]
    argument_regs = [
        Register(r)
        for r in [
            "r3",
            "r4",
            "r5",
            "r6",
            "r7",
            "r8",
            "r9",
            "r10",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "f8",
            "f9",
            "f10",
            "f11",
            "f12",
            "f13",
        ]
    ]
    simple_temp_regs = [Register(r) for r in ["r11", "r12"]]
    temp_regs = (
        argument_regs
        + simple_temp_regs
        + [
            Register(r)
            for r in [
                "r0",
                "f0",
                "cr0_gt",
                "cr0_lt",
                "cr0_eq",
                "cr0_so",
                "ctr",
            ]
        ]
    )
    saved_regs = [
        Register(r)
        for r in [
            # TODO: Some of the bits in CR are required to be saved (like cr2_gt)
            # When those bits are implemented, they should be added here
            "lr",
            # $r2 & $r13 are used for the small-data region, and are like $gp in MIPS
            "r2",
            "r13",
            "r14",
            "r15",
            "r16",
            "r17",
            "r18",
            "r19",
            "r20",
            "r21",
            "r22",
            "r23",
            "r24",
            "r25",
            "r26",
            "r27",
            "r28",
            "r29",
            "r30",
            "r31",
            "f14",
            "f15",
            "f16",
            "f17",
            "f18",
            "f19",
            "f20",
            "f21",
            "f22",
            "f23",
            "f24",
            "f25",
            "f26",
            "f27",
            "f28",
            "f29",
            "f30",
            "f31",
        ]
    ]
    all_regs = (
        saved_regs
        + temp_regs
        + [stack_pointer_reg]
        + [
            Register(r)
            for r in [
                # TODO: These `crX` registers are only used to parse instructions, but
                # the instructions that use these registers aren't implemented yet.
                "cr0",
                "cr1",
                "cr2",
                "cr3",
                "cr4",
                "cr5",
                "cr6",
                "cr7",
            ]
        ]
    )

    aliased_regs: Dict[str, Register] = {}

    @classmethod
    def missing_return(cls) -> List[Instruction]:
        return [cls.parse("blr", [], InstructionMeta.missing())]

    # List of all instructions where `$r0` as certian args is interpreted as `0`
    # instead of the contents of `$r0`. The dict value represents the argument
    # index that is affected.
    INSTRS_R0_AS_ZERO: ClassVar[Dict[str, int]] = {
        "addi": 1,
        "addis": 1,
        "dcbf": 0,
        "dcbi": 0,
        "dcbst": 0,
        "dcbt": 0,
        "dcbtst": 0,
        "dcbz": 0,
        "dcbz_l": 0,
        "eciwx": 1,
        "ecowx": 1,
        "icbi": 0,
        "lbz": 1,
        "lbzx": 1,
        "lfd": 1,
        "lfdx": 1,
        "lfs": 1,
        "lfsx": 1,
        "lha": 1,
        "lhax": 1,
        "lhbrx": 1,
        "lhz": 1,
        "lhzx": 1,
        "lmw": 1,
        "lswi": 1,
        "lswx": 1,
        "lwarx": 1,
        "lwbrx": 1,
        "lwz": 1,
        "lwzx": 1,
        "psq_lx": 1,
        "psq_stx": 1,
        "stb": 1,
        "stbx": 1,
        "stfd": 1,
        "stfdx": 1,
        "stfiwx": 1,
        "stfs": 1,
        "stfsx": 1,
        "sth": 1,
        "sthbrx": 1,
        "sthx": 1,
        "stmw": 1,
        "stswi": 1,
        "stswx": 1,
        "stw": 1,
        "stwbrx": 1,
        "stwcx.": 1,
        "stwx": 1,
    }

    @classmethod
    def normalize_instruction(cls, instr: AsmInstruction) -> AsmInstruction:
        # Remove +/- suffix, which indicates branch-(un)likely and can be ignored
        if instr.mnemonic.startswith("b") and (
            instr.mnemonic.endswith("+") or instr.mnemonic.endswith("-")
        ):
            return PpcArch.normalize_instruction(
                AsmInstruction(instr.mnemonic[:-1], instr.args)
            )

        args = instr.args
        r0_index = cls.INSTRS_R0_AS_ZERO.get(instr.mnemonic)
        if r0_index is not None and len(args) > r0_index:
            # If the argument at the given index is $r0, replace it with $zero
            r0_arg = args[r0_index]
            if r0_arg == Register("r0"):
                r0_arg = Register("zero")
            elif isinstance(r0_arg, AsmAddressMode) and r0_arg.rhs == Register("r0"):
                r0_arg = AsmAddressMode(lhs=r0_arg.lhs, rhs=Register("zero"))

            if r0_arg is not args[r0_index]:
                new_args = args[:]
                new_args[r0_index] = r0_arg
                return PpcArch.normalize_instruction(
                    AsmInstruction(instr.mnemonic, new_args)
                )
        if len(args) == 3:
            if (
                instr.mnemonic == "addi"
                and isinstance(args[2], Macro)
                and args[1] in (Register("r2"), Register("r13"))
                and args[2].macro_name in ("sda2", "sda21")
            ):
                return AsmInstruction("li", [args[0], args[2].argument])
        if len(args) == 2:
            if instr.mnemonic == "lis" and isinstance(args[1], AsmLiteral):
                lit = AsmLiteral((args[1].value & 0xFFFF) << 16)
                return AsmInstruction("li", [args[0], lit])
            if (
                instr.mnemonic == "lis"
                and isinstance(args[1], Macro)
                and args[1].macro_name == "ha"
                and isinstance(args[1].argument, AsmLiteral)
            ):
                # The @ha macro compensates for the sign bit of the corresponding @l
                value = args[1].argument.value
                if value & 0x8000:
                    value += 0x10000
                lit = AsmLiteral(value & 0xFFFF0000)
                return AsmInstruction("li", [args[0], lit])
            if instr.mnemonic.startswith("cmp"):
                # For the two-argument form of cmpw, the insert an implicit CR0 as the first arg
                cr0: Argument = Register("cr0")
                return AsmInstruction(instr.mnemonic, [cr0] + instr.args)
        return instr

    @classmethod
    def parse(
        cls, mnemonic: str, args: List[Argument], meta: InstructionMeta
    ) -> Instruction:
        inputs: List[Access] = []
        outputs: List[Access] = []
        clobbers: List[Access] = []
        jump_target: Optional[Union[JumpTarget, Register]] = None
        function_target: Optional[Union[AsmGlobalSymbol, Register]] = None
        is_conditional = False
        is_return = False

        cr0_bits: List[Access] = [
            Register("cr0_lt"),
            Register("cr0_gt"),
            Register("cr0_eq"),
            Register("cr0_so"),
        ]
        memory_sizes = {
            "b": 1,
            "h": 2,
            "w": 4,
            "mw": 4,
            "fs": 4,
            "fd": 8,
        }
        size = memory_sizes.get(mnemonic.lstrip("stl").rstrip("azux"))

        def make_memory_access(arg: AsmAddressMode) -> Access:
            assert isinstance(arg, AsmAddressMode)
            assert size is not None
            if arg.rhs == cls.stack_pointer_reg:
                return StackAccess(
                    offset=arg.lhs_as_literal(),
                    size=size,
                )
            return MemoryAccess(
                base_reg=arg.rhs,
                offset=arg.lhs,
                size=size,
            )

        if mnemonic == "blr":
            # Return
            inputs = [Register("lr")]
            is_return = True
        elif mnemonic in (
            "beqlr",
            "bgelr",
            "bgtlr",
            "blelr",
            "bltlr",
            "bnelr",
            "bnglr",
            "bnllr",
            "bnslr",
            "bsolr",
        ):
            # Conditional return
            inputs = cr0_bits + [Register("lr")]
            is_return = True
            is_conditional = True
        elif mnemonic == "bctr":
            # Jump table (switch)
            inputs = [Register("ctr")]
            jump_target = Register("ctr")
            is_conditional = True
        elif mnemonic == "bl":
            # Function call to label
            inputs = list(cls.argument_regs)
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            clobbers.append(MemoryAccess.arbitrary())
            assert isinstance(args[0], AsmGlobalSymbol)
            function_target = args[0]
        elif mnemonic == "bctrl":
            # Function call to pointer in $ctr
            inputs = list(cls.argument_regs)
            inputs.append(Register("clr"))
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            clobbers.append(MemoryAccess.arbitrary())
            function_target = Register("ctr")
        elif mnemonic == "blrl":
            # Function call to pointer in $lr
            inputs = list(cls.argument_regs)
            inputs.append(Register("lr"))
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            clobbers.append(MemoryAccess.arbitrary())
            function_target = Register("lr")
        elif mnemonic == "b":
            # Unconditional jump
            jump_target = get_jump_target(args[0])
        elif mnemonic in (
            "ble",
            "blt",
            "beq",
            "bge",
            "bgt",
            "bne",
            "bdnz",
            "bdz",
            "bdnz.fictive",
            "bdz.fictive",
        ):
            # Normal branch
            inputs = list(cr0_bits)
            jump_target = get_jump_target(args[-1])
            is_conditional = True
        elif mnemonic in cls.instrs_store:
            assert isinstance(args[0], Register)
            assert size is not None
            if mnemonic.endswith("x"):
                assert (
                    len(args) == 3
                    and isinstance(args[1], Register)
                    and isinstance(args[2], Register)
                )
                inputs = [args[0], args[1], args[2]]
                outputs = [MemoryAccess(args[1], args[2], size)]
            else:
                assert len(args) == 2 and isinstance(args[1], AsmAddressMode)
                inputs = [args[0], args[1].rhs]
                outputs = [make_memory_access(args[1])]
        elif mnemonic in cls.instrs_store_update:
            assert isinstance(args[0], Register)
            assert size is not None
            if mnemonic.endswith("x"):
                assert (
                    len(args) == 3
                    and isinstance(args[1], Register)
                    and isinstance(args[2], Register)
                )
                inputs = [args[0], args[1], args[2]]
                outputs = [MemoryAccess(args[1], args[2], size), args[1]]
            else:
                assert len(args) == 2 and isinstance(args[1], AsmAddressMode)
                inputs = [args[0], args[1].rhs]
                outputs = [make_memory_access(args[1]), args[1].rhs]
        elif mnemonic in cls.instrs_load_update:
            assert isinstance(args[0], Register)
            assert size is not None
            if mnemonic.endswith("x"):
                assert (
                    len(args) == 3
                    and isinstance(args[1], Register)
                    and isinstance(args[2], Register)
                )
                inputs = [MemoryAccess(args[1], args[2], size), args[1], args[2]]
                outputs = [args[0], args[1]]
            else:
                assert len(args) == 2 and isinstance(args[1], AsmAddressMode)
                inputs = [make_memory_access(args[1]), args[1].rhs]
                outputs = [args[0], args[1].rhs]
        elif mnemonic in ("stmw", "lmw"):
            assert (
                len(args) == 2
                and isinstance(args[0], Register)
                and isinstance(args[1], AsmAddressMode)
            )
            assert args[0].register_name[0] == "r"
            index = int(args[0].register_name[1:])
            offset = args[1].lhs_as_literal()
            while index <= 31:
                reg = Register(f"r{index}")
                mem = make_memory_access(
                    AsmAddressMode(rhs=args[1].rhs, lhs=AsmLiteral(offset))
                )
                if mnemonic == "stmw":
                    inputs.append(reg)
                    outputs.append(mem)
                else:
                    outputs.append(reg)
                    inputs.append(mem)
                index += 1
                offset += 4
            inputs.append(args[1].rhs)
        elif mnemonic in cls.instrs_no_dest:
            assert not any(isinstance(a, (Register, AsmAddressMode)) for a in args)
        elif mnemonic.rstrip(".") in cls.instrs_destination_first:
            assert isinstance(args[0], Register)
            outputs = [args[0]]
            if mnemonic.startswith("l") and size is not None:
                if mnemonic.endswith("x"):
                    assert (
                        len(args) == 3
                        and isinstance(args[1], Register)
                        and isinstance(args[2], Register)
                    )
                    inputs = [args[1], args[2], MemoryAccess(args[1], args[2], size)]
                else:
                    assert len(args) == 2 and isinstance(args[1], AsmAddressMode)
                    inputs = [args[1].rhs, make_memory_access(args[1])]
            elif mnemonic == "li" and isinstance(args[1], AsmAddressMode):
                # `li $rD, sym@sda21(r2)` is equivalent to `addi $rD, $r2, sym@sda21`
                assert len(args) == 2
                inputs = [args[1].rhs]
            else:
                assert not any(isinstance(a, AsmAddressMode) for a in args), (
                    mnemonic,
                    args,
                )
                inputs = [r for r in args[1:] if isinstance(r, Register)]
        elif mnemonic in cls.instrs_implicit_destination:
            assert len(args) == 1 and isinstance(args[0], Register)
            inputs = [args[0]]
            outputs = [cls.instrs_implicit_destination[mnemonic][0]]
        elif mnemonic in cls.instrs_ppc_compare:
            assert len(args) == 3 and isinstance(args[1], Register)
            inputs = [r for r in args[1:] if isinstance(r, Register)]
            outputs = list(cr0_bits)
        elif mnemonic in cls.instrs_ignore:
            # TODO: There might be some instrs to handle here
            pass
        elif args and isinstance(args[0], Register):
            # If the mnemonic is unsupported, guess
            inputs = [r for r in args[1:] if isinstance(r, Register)]
            outputs = [args[0]]

        if mnemonic.endswith("."):
            # PPC instructions ending in `.` update the condition reg
            outputs.extend(cr0_bits)

        return Instruction(
            mnemonic=mnemonic,
            args=args,
            meta=meta,
            inputs=inputs,
            outputs=outputs,
            clobbers=clobbers,
            jump_target=jump_target,
            function_target=function_target,
            is_conditional=is_conditional,
            is_return=is_return,
        )

    asm_patterns = [
        FcmpoCrorPattern(),
        TailCallPattern(),
        BoolCastPattern(),
        BranchCtrPattern(),
    ]

    instrs_ignore: InstrSet = {
        "nop",
        "b",
        # Assume stmw/lmw are only used for saving/restoring saved regs
        "stmw",
        "lmw",
        # `{crclr,crset} 6` are used as part of the ABI for floats & varargs
        # For now, we can ignore them (and later use them to help in function_abi)
        "crclr",
        "crset",
    }
    instrs_store: StoreInstrMap = {
        # Storage instructions
        "stb": lambda a: make_store(a, type=Type.int_of_size(8)),
        "sth": lambda a: make_store(a, type=Type.int_of_size(16)),
        "stw": lambda a: make_store(a, type=Type.reg32(likely_float=False)),
        "stbx": lambda a: make_storex(a, type=Type.int_of_size(8)),
        "sthx": lambda a: make_storex(a, type=Type.int_of_size(16)),
        "stwx": lambda a: make_storex(a, type=Type.reg32(likely_float=False)),
        # TODO: Do we need to model the truncation from f64 to f32 here?
        "stfs": lambda a: make_store(a, type=Type.f32()),
        "stfd": lambda a: make_store(a, type=Type.f64()),
        "stfsx": lambda a: make_storex(a, type=Type.f32()),
        "stfdx": lambda a: make_storex(a, type=Type.f64()),
    }
    instrs_store_update: StoreInstrMap = {
        "stbu": lambda a: make_store(a, type=Type.int_of_size(8)),
        "sthu": lambda a: make_store(a, type=Type.int_of_size(16)),
        "stwu": lambda a: make_store(a, type=Type.reg32(likely_float=False)),
        "stbux": lambda a: make_storex(a, type=Type.int_of_size(8)),
        "sthux": lambda a: make_storex(a, type=Type.int_of_size(16)),
        "stwux": lambda a: make_storex(a, type=Type.reg32(likely_float=False)),
        "stfsu": lambda a: make_store(a, type=Type.f32()),
        "stfdu": lambda a: make_store(a, type=Type.f64()),
        "stfsux": lambda a: make_storex(a, type=Type.f32()),
        "stfdux": lambda a: make_storex(a, type=Type.f64()),
    }
    instrs_branches: CmpInstrMap = {
        # Branch instructions/pseudoinstructions
        # Technically `bge` is defined as `cr0_gt || cr0_eq`; not as `!cr0_lt`
        # This assumption may not hold if the bits are modified with instructions like
        # `crand` which modify individual bits in CR.
        "beq": lambda a: a.cmp_reg("cr0_eq"),
        "bge": lambda a: a.cmp_reg("cr0_lt").negated(),
        "bgt": lambda a: a.cmp_reg("cr0_gt"),
        "ble": lambda a: a.cmp_reg("cr0_gt").negated(),
        "blt": lambda a: a.cmp_reg("cr0_lt"),
        "bne": lambda a: a.cmp_reg("cr0_eq").negated(),
        "bns": lambda a: a.cmp_reg("cr0_so").negated(),
        "bso": lambda a: a.cmp_reg("cr0_so"),
        "bdnz.fictive": lambda a: a.cmp_reg("ctr"),
        "bdz.fictive": lambda a: a.cmp_reg("ctr").negated(),
    }
    instrs_float_branches: InstrSet = {}
    instrs_jumps: InstrSet = {
        # Unconditional jumps
        "b",
        "blr",
        "bctr",
    }
    instrs_fn_call: InstrSet = {
        # Function call
        "bl",
        "blrl",
        "bctrl",
    }
    instrs_no_dest: StmtInstrMap = {
        "sync": lambda a: void_fn_op("MIPS2C_SYNC", []),
        "isync": lambda a: void_fn_op("MIPS2C_SYNC", []),
    }
    instrs_destination_first: InstrMap = {
        # Integer arithmetic
        # TODO: Read XER_CA in extended instrs, instead of using CarryBit
        "add": lambda a: handle_add(a),
        "addc": lambda a: handle_add(a),
        "adde": lambda a: CarryBit.add_to(handle_add(a)),
        "addze": lambda a: CarryBit.add_to(a.reg(1)),
        "addi": lambda a: handle_addi(a),
        "addic": lambda a: handle_addi(a),
        "addis": lambda a: handle_addis(a),
        "subf": lambda a: fold_divmod(
            BinaryOp.intptr(left=a.reg(2), op="-", right=a.reg(1))
        ),
        "subfc": lambda a: fold_divmod(
            BinaryOp.intptr(left=a.reg(2), op="-", right=a.reg(1))
        ),
        "subfe": lambda a: CarryBit.sub_from(
            fold_divmod(BinaryOp.intptr(left=a.reg(2), op="-", right=a.reg(1)))
        ),
        "subfic": lambda a: fold_divmod(
            BinaryOp.intptr(left=a.imm(2), op="-", right=a.reg(1))
        ),
        "subfze": lambda a: CarryBit.sub_from(
            fold_mul_chains(
                UnaryOp(op="-", expr=as_s32(a.reg(1), silent=True), type=Type.s32())
            )
        ),
        "neg": lambda a: fold_mul_chains(
            UnaryOp(op="-", expr=as_s32(a.reg(1), silent=True), type=Type.s32())
        ),
        "divw": lambda a: BinaryOp.s32(a.reg(1), "/", a.reg(2)),
        "divwu": lambda a: BinaryOp.u32(a.reg(1), "/", a.reg(2)),
        "mulli": lambda a: BinaryOp.int(a.reg(1), "*", a.imm(2)),
        "mullw": lambda a: BinaryOp.int(a.reg(1), "*", a.reg(2)),
        "mulhw": lambda a: fold_divmod(BinaryOp.int(a.reg(1), "MULT_HI", a.reg(2))),
        "mulhwu": lambda a: fold_divmod(BinaryOp.int(a.reg(1), "MULTU_HI", a.reg(2))),
        # Bit arithmetic
        "or": lambda a: handle_or(a.reg(1), a.reg(2)),
        "ori": lambda a: handle_or(a.reg(1), a.unsigned_imm(2)),
        "oris": lambda a: handle_or(a.reg(1), a.shifted_imm(2)),
        "and": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.reg(2)),
        "andc": lambda a: BinaryOp.int(
            left=a.reg(1), op="&", right=UnaryOp("~", a.reg(2), type=Type.intish())
        ),
        "not": lambda a: UnaryOp("~", a.reg(1), type=Type.intish()),
        "nor": lambda a: UnaryOp(
            "~", BinaryOp.int(left=a.reg(1), op="|", right=a.reg(2)), type=Type.intish()
        ),
        "xor": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.reg(2)),
        "eqv": lambda a: UnaryOp(
            "~", BinaryOp.int(left=a.reg(1), op="^", right=a.reg(2)), type=Type.intish()
        ),
        "andi": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.unsigned_imm(2)),
        "andis": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.shifted_imm(2)),
        "xori": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.unsigned_imm(2)),
        "xoris": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.shifted_imm(2)),
        "boolcast.fictive": lambda a: UnaryOp(
            op="!!", expr=a.reg(1), type=Type.intish()
        ),
        "rlwimi": lambda a: handle_rlwimi(
            a.reg(0), a.reg(1), a.imm_value(2), a.imm_value(3), a.imm_value(4)
        ),
        "rlwinm": lambda a: handle_rlwinm(
            a.reg(1), a.imm_value(2), a.imm_value(3), a.imm_value(4)
        ),
        "extlwi": lambda a: handle_rlwinm(
            a.reg(1), a.imm_value(3), 0, a.imm_value(2) - 1
        ),
        "extrwi": lambda a: handle_rlwinm(
            a.reg(1), a.imm_value(3) + a.imm_value(2), 32 - a.imm_value(2), 31
        ),
        "rotlwi": lambda a: handle_rlwinm(a.reg(1), a.imm_value(2), 0, 31),
        "rotrwi": lambda a: handle_rlwinm(a.reg(1), 32 - a.imm_value(2), 0, 31),
        "slwi": lambda a: handle_rlwinm(
            a.reg(1), a.imm_value(2), 0, 31 - a.imm_value(2)
        ),
        "srwi": lambda a: handle_rlwinm(
            a.reg(1), 32 - a.imm_value(2), a.imm_value(2), 31
        ),
        "clrlwi": lambda a: handle_rlwinm(a.reg(1), 0, a.imm_value(2), 31),
        "clrrwi": lambda a: handle_rlwinm(a.reg(1), 0, 0, 31 - a.imm_value(2)),
        "clrlslwi": lambda a: handle_rlwinm(
            a.reg(1),
            a.imm_value(3),
            a.imm_value(2) - a.imm_value(3),
            31 - a.imm_value(3),
        ),
        "slw": lambda a: fold_mul_chains(
            BinaryOp.int(left=a.reg(1), op="<<", right=as_intish(a.reg(2)))
        ),
        "srw": lambda a: fold_divmod(
            BinaryOp(
                left=as_u32(a.reg(1)),
                op=">>",
                right=as_intish(a.reg(2)),
                type=Type.u32(),
            )
        ),
        "sraw": lambda a: fold_divmod(
            BinaryOp(
                left=as_s32(a.reg(1)),
                op=">>",
                right=as_intish(a.reg(2)),
                type=Type.s32(),
            )
        ),
        "srawi": lambda a: handle_sra(a),
        "extsb": lambda a: handle_convert(a.reg(1), Type.s8(), Type.intish()),
        "extsh": lambda a: handle_convert(a.reg(1), Type.s16(), Type.intish()),
        "cntlzw": lambda a: UnaryOp(op="CLZ", expr=a.reg(1), type=Type.intish()),
        # Integer Loads
        "lba": lambda a: handle_load(a, type=Type.s8()),
        "lbz": lambda a: handle_load(a, type=Type.u8()),
        "lha": lambda a: handle_load(a, type=Type.s16()),
        "lhz": lambda a: handle_load(a, type=Type.u16()),
        "lwz": lambda a: handle_load(a, type=Type.reg32(likely_float=False)),
        "lbax": lambda a: handle_loadx(a, type=Type.s8()),
        "lbzx": lambda a: handle_loadx(a, type=Type.u8()),
        "lhax": lambda a: handle_loadx(a, type=Type.s16()),
        "lhzx": lambda a: handle_loadx(a, type=Type.u16()),
        "lwzx": lambda a: handle_loadx(a, type=Type.reg32(likely_float=False)),
        # Load Immediate
        "li": lambda a: a.full_imm(1),
        "lis": lambda a: load_upper(a),
        # Move from Special Register
        "mflr": lambda a: a.regs[Register("lr")],
        "mfctr": lambda a: a.regs[Register("ctr")],
        "mr": lambda a: a.reg(1),
        # Floating Point Loads
        # TODO: Do we need to model the promotion from f32 to f64 here?
        "lfs": lambda a: handle_load(a, type=Type.f32()),
        "lfd": lambda a: handle_load(a, type=Type.f64()),
        "lfsx": lambda a: handle_loadx(a, type=Type.f32()),
        "lfdx": lambda a: handle_loadx(a, type=Type.f64()),
        # Floating Point Arithmetic
        "fadd": lambda a: handle_add_double(a),
        "fadds": lambda a: handle_add_float(a),
        "fdiv": lambda a: BinaryOp.f64(a.reg(1), "/", a.reg(2)),
        "fdivs": lambda a: BinaryOp.f32(a.reg(1), "/", a.reg(2)),
        "fmul": lambda a: BinaryOp.f64(a.reg(1), "*", a.reg(2)),
        "fmuls": lambda a: BinaryOp.f32(a.reg(1), "*", a.reg(2)),
        "fsub": lambda a: BinaryOp.f64(a.reg(1), "-", a.reg(2)),
        "fsubs": lambda a: BinaryOp.f32(a.reg(1), "-", a.reg(2)),
        "fneg": lambda a: UnaryOp(op="-", expr=a.reg(1), type=Type.floatish()),
        "fmr": lambda a: a.reg(1),
        "frsp": lambda a: handle_convert(a.reg(1), Type.f32(), Type.f64()),
        # TODO: This yields some awkward-looking C code, often in the form:
        # `sp100 = (bitwise f64) (s32) x; y = sp104;` instead of `y = (s32) x;`.
        # We should try to detect these idioms, along with int-to-float
        "fctiwz": lambda a: handle_convert(a.reg(1), Type.s32(), Type.floatish()),
        # Floating Poing Fused Multiply-{Add,Sub}
        "fmadd": lambda a: BinaryOp.f64(
            BinaryOp.f64(a.reg(1), "*", a.reg(2)), "+", a.reg(3)
        ),
        "fmadds": lambda a: BinaryOp.f32(
            BinaryOp.f32(a.reg(1), "*", a.reg(2)), "+", a.reg(3)
        ),
        "fnmadd": lambda a: UnaryOp(
            op="-",
            expr=BinaryOp.f64(BinaryOp.f64(a.reg(1), "*", a.reg(2)), "+", a.reg(3)),
            type=Type.f64(),
        ),
        "fnmadds": lambda a: UnaryOp(
            op="-",
            expr=BinaryOp.f32(BinaryOp.f32(a.reg(1), "*", a.reg(2)), "+", a.reg(3)),
            type=Type.f32(),
        ),
        "fmsub": lambda a: BinaryOp.f64(
            BinaryOp.f64(a.reg(1), "*", a.reg(2)), "-", a.reg(3)
        ),
        "fmsubs": lambda a: BinaryOp.f32(
            BinaryOp.f32(a.reg(1), "*", a.reg(2)), "-", a.reg(3)
        ),
        "fnmsub": lambda a: UnaryOp(
            op="-",
            expr=BinaryOp.f64(BinaryOp.f64(a.reg(1), "*", a.reg(2)), "-", a.reg(3)),
            type=Type.f64(),
        ),
        "fnmsubs": lambda a: UnaryOp(
            op="-",
            expr=BinaryOp.f32(BinaryOp.f32(a.reg(1), "*", a.reg(2)), "-", a.reg(3)),
            type=Type.f32(),
        ),
        # TODO: Detect if we should use fabs or fabsf
        "fabs": lambda a: fn_op("fabs", [a.reg(1)], Type.floatish()),
        "fres": lambda a: fn_op("__fres", [a.reg(1)], Type.floatish()),
        "frsqrte": lambda a: fn_op("__frsqrte", [a.reg(1)], Type.floatish()),
        "fsel": lambda a: TernaryOp(
            cond=BinaryOp.fcmp(a.reg(1), ">=", Literal(0)),
            left=a.reg(2),
            right=a.reg(3),
            type=Type.floatish(),
        ),
    }
    instrs_load_update: InstrMap = {
        "lbau": lambda a: handle_load(a, type=Type.s8()),
        "lbzu": lambda a: handle_load(a, type=Type.u8()),
        "lhau": lambda a: handle_load(a, type=Type.s16()),
        "lhzu": lambda a: handle_load(a, type=Type.u16()),
        "lwzu": lambda a: handle_load(a, type=Type.reg32(likely_float=False)),
        "lbaux": lambda a: handle_loadx(a, type=Type.s8()),
        "lbzux": lambda a: handle_loadx(a, type=Type.u8()),
        "lhaux": lambda a: handle_loadx(a, type=Type.s16()),
        "lhzux": lambda a: handle_loadx(a, type=Type.u16()),
        "lwzux": lambda a: handle_loadx(a, type=Type.reg32(likely_float=False)),
        "lfsu": lambda a: handle_load(a, type=Type.f32()),
        "lfdu": lambda a: handle_load(a, type=Type.f64()),
        "lfsux": lambda a: handle_loadx(a, type=Type.f32()),
        "lfdux": lambda a: handle_loadx(a, type=Type.f64()),
    }

    instrs_implicit_destination: ImplicitInstrMap = {
        "mtlr": (Register("lr"), lambda a: a.reg(0)),
        "mtctr": (Register("ctr"), lambda a: a.reg(0)),
    }

    instrs_ppc_compare: PpcCmpInstrMap = {
        # Integer (signed/unsigned)
        "cmpw": lambda a, op: BinaryOp.sintptr_cmp(a.reg(1), op, a.reg(2)),
        "cmpwi": lambda a, op: BinaryOp.sintptr_cmp(a.reg(1), op, a.imm(2)),
        "cmplw": lambda a, op: BinaryOp.uintptr_cmp(a.reg(1), op, a.reg(2)),
        "cmplwi": lambda a, op: BinaryOp.uintptr_cmp(a.reg(1), op, a.imm(2)),
        # Floating point
        # TODO: There is a difference in how these two instructions handle NaN
        "fcmpo": lambda a, op: BinaryOp.fcmp(a.reg(1), op, a.reg(2)),
        "fcmpu": lambda a, op: BinaryOp.fcmp(a.reg(1), op, a.reg(2)),
        "fcmpo.lte.fictive": lambda a, op: BinaryOp.fcmp(
            a.reg(1), op if op != "==" else "<=", a.reg(2)
        ),
        "fcmpo.gte.fictive": lambda a, op: BinaryOp.fcmp(
            a.reg(1), op if op != "==" else ">=", a.reg(2)
        ),
    }

    @staticmethod
    def function_abi(
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> Abi:
        known_slots: List[AbiArgSlot] = []
        candidate_slots: List[AbiArgSlot] = []

        # $rX & $fX regs can be interspersed in function args, unlike in the MIPS O32 ABI
        intptr_regs = [r for r in PpcArch.argument_regs if r.register_name[0] != "f"]
        float_regs = [r for r in PpcArch.argument_regs if r.register_name[0] == "f"]

        if fn_sig.params_known:
            for ind, param in enumerate(fn_sig.params):
                # TODO: Support passing parameters on the stack
                param_type = param.type.decay()
                reg: Optional[Register]
                try:
                    if param_type.is_float():
                        reg = float_regs.pop(0)
                    else:
                        reg = intptr_regs.pop(0)
                except IndexError:
                    # Stack variable
                    reg = None
                known_slots.append(
                    AbiArgSlot(
                        offset=4 * ind, reg=reg, name=param.name, type=param_type
                    )
                )
            if fn_sig.is_variadic:
                # TODO: Find a better value to use for `offset`?
                for reg in intptr_regs:
                    candidate_slots.append(
                        AbiArgSlot(
                            offset=4 * len(known_slots), reg=reg, type=Type.intptr()
                        )
                    )
                for reg in float_regs:
                    candidate_slots.append(
                        AbiArgSlot(
                            offset=4 * len(known_slots), reg=reg, type=Type.floatish()
                        )
                    )
        else:
            for ind, reg in enumerate(PpcArch.argument_regs):
                if reg.register_name[0] != "f":
                    candidate_slots.append(
                        AbiArgSlot(offset=4 * ind, reg=reg, type=Type.intptr())
                    )
                else:
                    candidate_slots.append(
                        AbiArgSlot(offset=4 * ind, reg=reg, type=Type.floatish())
                    )

        valid_extra_regs: Set[Register] = {
            slot.reg for slot in known_slots if slot.reg is not None
        }
        possible_slots: List[AbiArgSlot] = []
        for slot in candidate_slots:
            if slot.reg is None or slot.reg not in likely_regs:
                continue

            # Don't pass this register if lower numbered ones are undefined.
            if slot == candidate_slots[0]:
                # For varargs, a subset of regs may be used. Don't check
                # earlier registers for the first member of that subset.
                pass
            else:
                # Only r3-r10/f1-f13 can be used for arguments
                regname = slot.reg.register_name
                prev_reg = Register(f"{regname[0]}{int(regname[1:])-1}")
                if (
                    prev_reg in PpcArch.argument_regs
                    and prev_reg not in valid_extra_regs
                ):
                    continue

            valid_extra_regs.add(slot.reg)

            # Skip registers that are untouched from the initial parameter
            # list. This is sometimes wrong (can give both false positives
            # and negatives), but having a heuristic here is unavoidable
            # without access to function signatures, or when dealing with
            # varargs functions. Decompiling multiple functions at once
            # would help.
            # TODO: don't do this in the middle of the argument list,
            # except for f12 if a0 is passed and such.
            if not likely_regs[slot.reg]:
                continue

            possible_slots.append(slot)

        return Abi(
            arg_slots=known_slots,
            possible_slots=possible_slots,
        )

    @staticmethod
    def function_return(expr: Expression) -> Dict[Register, Expression]:
        return {
            Register("f1"): Cast(
                expr, reinterpret=True, silent=True, type=Type.floatish()
            ),
            Register("r3"): Cast(
                expr, reinterpret=True, silent=True, type=Type.intptr()
            ),
            Register("r4"): as_u32(
                Cast(expr, reinterpret=True, silent=False, type=Type.u64())
            ),
        }
