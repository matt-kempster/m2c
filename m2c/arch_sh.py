from __future__ import annotations
from dataclasses import replace
from typing import Callable, Dict, List, Optional

from .error import DecompFailure
from .options import Target
from .asm_file import AsmSymbolicData
from .asm_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    AsmState,
    BinOp,
    JumpTarget,
    Register,
    Writeback,
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
)
from .translate import (
    Abi,
    AbiArgSlot,
    Arch,
    ArgLoc,
    BinaryOp,
    Cast,
    ErrorExpr,
    ExprStmt,
    Expression,
    InstrMap,
    InstrArgs,
    Literal,
    NodeState,
    UnaryOp,
    as_s16,
    as_type,
    as_u16,
    as_u32,
)

from .evaluate import (
    condition_from_expr,
    fn_op,
    fold_mul_chains,
    fold_shift_right,
    handle_add,
    handle_addi,
    handle_bitinv,
    handle_load,
    handle_loadx,
    handle_or,
    handle_sub,
    handle_xor,
    make_store,
    make_storex,
    void_fn_op,
)

from .types import FunctionSignature, Type


class JumpTablePattern(SimpleAsmPattern):
    pattern = make_pattern(
        "mov $x, $i",
        "add $i, $i",
        "mova _, $b",
        "mov.w @($b,$i),$i",
        "add $i, $b",
        "jmp @$b",
        "nop",
    )

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        mova = m.body[2]
        assert isinstance(mova, Instruction)
        assert isinstance(mova.args[0], AsmGlobalSymbol)

        table_name = mova.args[0].symbol_name
        table = m.asm_data.values.get(table_name)
        if table is None:
            return None
        targets: List[AsmGlobalSymbol] = []
        for entry in table.data:
            if (
                not isinstance(entry, AsmSymbolicData)
                or not isinstance(entry.data, BinOp)
                or entry.data.op != "-"
                or not isinstance(entry.data.lhs, AsmGlobalSymbol)
                or entry.data.rhs != AsmGlobalSymbol(table_name)
            ):
                return None
            targets.append(entry.data.lhs)
        if not targets:
            return None
        return Replacement(
            [
                AsmInstruction("tablejmp.fictive", [m.regs["x"], *targets]),
                AsmInstruction("nop", []),
            ],
            len(m.body),
        )


class NegateTPattern(SimpleAsmPattern):
    pattern = make_pattern(
        "rotcl $r",
        "xor #1, $r",
        "rotcr $r",
    )

    def replace(self, m: AsmMatch) -> Replacement:
        return Replacement([AsmInstruction("negatet.fictive", [])], len(m.body))


class SubcSelfPattern(SimpleAsmPattern):
    pattern = make_pattern("subc $r, $r")

    def replace(self, m: AsmMatch) -> Replacement:
        reg = m.regs["r"]
        return Replacement(
            [
                AsmInstruction("movt", [reg]),
                AsmInstruction("neg", [reg, reg]),
            ],
            len(m.body),
        )


class Sh2AddrModeWritebackPattern(AsmPattern):
    """Replace writebacks in mov address modes by separate add instructions."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        instr = matcher.input[matcher.index]
        if not isinstance(instr, Instruction) or not instr.args:
            return None
        if instr.mnemonic not in ("mov.b", "mov.w", "mov.l"):
            return None

        if isinstance(instr.args[0], AsmAddressMode):
            addr = instr.args[0]
            addr_arg = 0
        elif isinstance(instr.args[1], AsmAddressMode):
            addr = instr.args[1]
            addr_arg = 1
        else:
            return None

        if addr.writeback is None:
            return None
        if addr.base == Sh2Arch.stack_pointer_reg:
            return None

        if addr.base in (instr.args[0], instr.args[1]):
            raise DecompFailure(
                "Writeback with base register also used as load/store value, "
                f"for instruction: {instr}"
            )

        assert addr.addend == AsmLiteral(0)

        new_args = list(instr.args)
        new_args[addr_arg] = replace(addr, writeback=None)
        stride = Sh2Arch.mov_stride(instr.mnemonic)
        if addr.writeback == Writeback.PRE:
            return Replacement(
                [
                    AsmInstruction("add", [AsmLiteral(-stride), addr.base]),
                    AsmInstruction(instr.mnemonic, new_args),
                ],
                1,
            )

        return Replacement(
            [
                AsmInstruction(instr.mnemonic, new_args),
                AsmInstruction("add", [AsmLiteral(stride), addr.base]),
            ],
            1,
        )


class DivisionHelperPattern(SimpleAsmPattern):
    pattern = make_pattern("mov.l _, $t", "jsr @$t", "*")

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        load = m.body[0]
        delay_slot = m.wildcard_items[0]
        assert isinstance(load, Instruction)
        if not isinstance(load.args[0], AsmGlobalSymbol):
            return None
        if not isinstance(delay_slot, Instruction):
            return None

        entry = m.asm_data.values.get(load.args[0].symbol_name)
        if entry is None:
            return None
        target = entry.data_at_offset(0, 4)
        if not isinstance(target, AsmSymbolicData):
            return None
        target_name = target.as_symbol_without_addend()
        if target_name is None:
            return None
        mnemonic = {
            "___sdivsi3": "sdiv.fictive",
            "___udivsi3": "udiv.fictive",
        }.get(target_name)
        if mnemonic is None:
            return None
        division = AsmInstruction(mnemonic, [Register("r4"), Register("r5")])
        return Replacement([delay_slot, division], 3)


class Sh2Arch(Arch):
    arch = Target.ArchEnum.SH2

    def c_symbol_name(self, asm_name: str) -> str:
        if (
            len(asm_name) >= 2
            and asm_name[0] == "_"
            and (asm_name[1].isalpha() or asm_name[1] == "_")
        ):
            return asm_name[1:]
        return asm_name

    re_comment = r"!.*"
    supports_dollar_regs = False
    supports_at_addressing = True
    asm_word_size = 2

    has_delay_slots = True

    home_space_size = 0

    stack_pointer_reg = Register("r15")
    frame_pointer_regs = [Register("r14")]
    return_address_reg = Register("pr")

    base_return_regs = [(Register("r0"), False)]
    all_return_regs = [Register("r0"), Register("r1")]

    argument_regs = [Register(r) for r in ["r4", "r5", "r6", "r7"]]
    simple_temp_regs = [Register(r) for r in ["r0", "r1", "r2", "r3"]]
    temp_regs = argument_regs + simple_temp_regs + [Register("condition_bit")]

    saved_regs = [
        Register(r) for r in ["r8", "r9", "r10", "r11", "r12", "r13", "r14", "pr"]
    ]

    all_regs = (
        saved_regs
        + temp_regs
        + [stack_pointer_reg]
        + [
            Register(r)
            for r in [
                "mach",
                "macl",
                "sr",
                "gbr",
                "vbr",
            ]
        ]
    )

    @staticmethod
    def mov_type(mnemonic: str) -> Type:
        return {
            "mov.b": Type.s8(),
            "mov.w": Type.s16(),
            "mov.l": Type.reg32(likely_float=False),
        }[mnemonic]

    @staticmethod
    def mov_stride(mnemonic: str) -> int:
        return {
            "mov.b": 1,
            "mov.w": 2,
            "mov.l": 4,
        }[mnemonic]

    aliased_regs: Dict[str, Register] = {}

    @classmethod
    def missing_return(cls) -> List[Instruction]:
        meta = InstructionMeta.missing()
        return [
            cls.parse("rts", [], meta),
            cls.parse("nop", [], meta),
        ]

    @classmethod
    def normalize_instruction(
        cls, instr: AsmInstruction, asm_state: AsmState
    ) -> AsmInstruction:
        mnemonic = {
            "muls.w": "muls",
            "mulu.w": "mulu",
        }.get(instr.mnemonic, instr.mnemonic)
        return replace(instr, mnemonic=mnemonic)

    @classmethod
    def parse(
        cls, mnemonic: str, args: List[Argument], meta: InstructionMeta
    ) -> Instruction:
        inputs: List[Location] = []
        clobbers: List[Location] = []
        outputs: List[Location] = []
        is_return = False
        is_load = False
        is_store = False
        has_delay_slot = False
        is_conditional = False
        jump_target = None
        function_target: Optional[Argument] = None
        eval_fn: Optional[Callable[[NodeState, InstrArgs], object]] = None

        if mnemonic == "rts":
            assert len(args) == 0
            inputs = [Register("pr")]
            is_return = True
            has_delay_slot = True
        elif mnemonic == "nop":
            assert len(args) == 0
        elif mnemonic in ("mov", "lds", "sts"):
            assert len(args) == 2 and isinstance(args[1], Register)
            outputs = [args[1]]
            if isinstance(args[0], Register):
                inputs = [args[0]]
                eval_fn = lambda s, a: s.set_reg(a.reg_ref(1), a.reg(0))
            else:
                assert isinstance(args[0], AsmLiteral)
                eval_fn = lambda s, a: s.set_reg(a.reg_ref(1), a.s8_imm(0))

        elif mnemonic == "mova":
            assert (
                len(args) == 2
                and isinstance(args[0], AsmGlobalSymbol)
                and args[1] == Register("r0")
            )
            outputs = [Register("r0")]
            eval_fn = lambda s, a: s.set_reg(Register("r0"), a.full_imm(0))
        elif mnemonic in ("mov.b", "mov.l", "mov.w"):
            assert len(args) == 2
            if isinstance(args[0], Register):
                assert isinstance(args[1], AsmAddressMode)
                inputs = [args[0], args[1].base]
                if isinstance(args[1].addend, Register):
                    inputs.append(args[1].addend)
                is_store = True
                if args[1].writeback is None:

                    def eval_fn(s: NodeState, a: InstrArgs) -> None:
                        store_type = cls.mov_type(mnemonic)
                        address = a.raw_arg(1)
                        assert isinstance(address, AsmAddressMode)
                        if isinstance(address.addend, Register):
                            store = make_storex(
                                replace(
                                    a,
                                    raw_args=[
                                        a.raw_arg(0),
                                        address.base,
                                        address.addend,
                                    ],
                                ),
                                store_type,
                            )
                        else:
                            store = make_store(a, store_type)
                        if store is not None:
                            s.store_memory(store, a.reg_ref(0))

                elif args[1].base == cls.stack_pointer_reg:
                    assert args[1].writeback == Writeback.PRE

                    def eval_fn(s: NodeState, a: InstrArgs) -> None:
                        source_raw = a.regs.get_raw(a.reg_ref(0))
                        assert source_raw is not None
                        if not s.stack_info.should_save(source_raw, None):
                            s.push_subroutine_arg(a.reg(0))

                else:
                    # writeback on other registers is rewritten into an explicit
                    # add by Sh2AddrModeWritebackPattern, so no eval_fn is needed.
                    pass
            else:
                assert isinstance(args[1], Register)
                if isinstance(args[0], AsmAddressMode):
                    # We intentionally ignore args[0].writeback. It can only be non-None
                    # for stack pointer writes (Sh2AddrModeWritebackPattern gets rid of
                    # it for other registers), and then only during the epilogue, which
                    # we don't emit any code for.
                    inputs = [args[0].base]
                    if isinstance(args[0].addend, Register):
                        inputs.append(args[0].addend)
                outputs = [args[1]]
                is_load = True
                load_type = cls.mov_type(mnemonic)
                if isinstance(args[0], AsmAddressMode) and isinstance(
                    args[0].addend, Register
                ):
                    indexed_base = args[0].base
                    indexed_addend = args[0].addend
                    eval_fn = lambda s, a: s.set_reg(
                        a.reg_ref(1),
                        handle_loadx(
                            replace(
                                a,
                                raw_args=[
                                    a.raw_arg(1),
                                    indexed_base,
                                    indexed_addend,
                                ],
                            ),
                            type=load_type,
                        ),
                    )
                else:
                    eval_fn = lambda s, a: s.set_reg(
                        a.reg_ref(1),
                        handle_load(
                            replace(a, raw_args=[a.raw_arg(1), a.raw_arg(0)]),
                            type=load_type,
                        ),
                    )
        elif mnemonic in ("sts.l", "stc.l"):
            assert (
                len(args) == 2
                and isinstance(args[0], Register)
                and isinstance(args[1], AsmAddressMode)
                and args[1].writeback == Writeback.PRE
            )
            mem = args[1]
            inputs = [args[0], mem.base]
            is_store = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                assert args[0] == Register("pr")
                assert mem.base == cls.stack_pointer_reg

        elif mnemonic in ("lds.l", "ldc.l"):
            assert (
                len(args) == 2
                and isinstance(args[0], AsmAddressMode)
                and args[0].writeback == Writeback.POST
                and isinstance(args[1], Register)
            )
            mem = args[0]
            inputs = [mem.base]
            outputs = [args[1]]
            is_load = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                assert args[1] == Register("pr")
                assert mem.base == cls.stack_pointer_reg

        elif mnemonic in cls.instrs_read_modify_write:
            assert len(args) == 2 and isinstance(args[1], Register)
            inputs = [args[1]]
            if isinstance(args[0], Register):
                inputs.insert(0, args[0])
            outputs = [args[1]]
            eval_fn = lambda s, a: s.set_reg(
                a.reg_ref(1), cls.instrs_read_modify_write[mnemonic](a)
            )
        elif mnemonic in cls.instrs_source_dest:
            assert (
                len(args) == 2
                and isinstance(args[0], Register)
                and isinstance(args[1], Register)
            )
            inputs = [args[0]]
            outputs = [args[1]]
            eval_fn = lambda s, a: s.set_reg(
                a.reg_ref(1), cls.instrs_source_dest[mnemonic](a)
            )
        elif mnemonic in cls.instrs_multiply:
            assert (
                len(args) == 2
                and isinstance(args[0], Register)
                and isinstance(args[1], Register)
            )
            inputs = [args[0], args[1]]
            outputs = [Register("macl")]
            eval_fn = lambda s, a: s.set_reg(
                Register("macl"), cls.instrs_multiply[mnemonic](a)
            )
        elif mnemonic in ("dmuls.l", "dmulu.l"):
            assert (
                len(args) == 2
                and isinstance(args[0], Register)
                and isinstance(args[1], Register)
            )
            inputs = [args[0], args[1]]
            outputs = [Register("mach"), Register("macl")]
            signed = mnemonic == "dmuls.l"

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                # 64-bit, mach has high part, macl has low part
                operand_type = Type.s32() if signed else Type.u32()
                product_type = Type.s64() if signed else Type.u64()
                lhs = as_type(a.reg(1), operand_type, silent=True)
                rhs = as_type(a.reg(0), operand_type, silent=True)
                product = BinaryOp(
                    as_type(lhs, product_type, silent=False),
                    "*",
                    as_type(rhs, product_type, silent=False),
                    type=product_type,
                )
                high = BinaryOp(
                    product,
                    ">>",
                    Literal(32),
                    type=product_type,
                )
                s.set_reg(
                    Register("mach"),
                    as_type(high, operand_type, silent=True),
                )
                s.set_reg(
                    Register("macl"),
                    as_type(product, Type.u32(), silent=True),
                )

        elif mnemonic == "clrmac":
            assert not args
            outputs = [Register("macl"), Register("mach")]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(Register("macl"), Literal(0))
                s.set_reg(Register("mach"), Literal(0))

        elif mnemonic in cls.instrs_shift:
            assert len(args) == 1 and isinstance(args[0], Register)
            inputs = [args[0]]
            outputs = [args[0]]
            if mnemonic in ("shll", "shlr", "shar", "rotl", "rotr"):
                outputs.append(Register("condition_bit"))

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                original = a.reg(0)
                if mnemonic in ("shlr", "shar", "rotr"):
                    carry = BinaryOp.intptr(original, "&", Literal(1))
                else:
                    carry = BinaryOp.uint(original, ">>", Literal(31))
                if mnemonic in ("shll", "shlr", "shar", "rotl", "rotr"):
                    s.set_reg(Register("condition_bit"), carry)
                s.set_reg(a.reg_ref(0), cls.instrs_shift[mnemonic](a))

        elif mnemonic == "dt":
            assert len(args) == 1 and isinstance(args[0], Register)
            inputs = [args[0]]
            outputs = [args[0], Register("condition_bit")]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                value = s.set_reg(
                    a.reg_ref(0),
                    BinaryOp.intptr(a.reg(0), "-", Literal(1)),
                )
                s.set_reg(
                    Register("condition_bit"),
                    BinaryOp.icmp(value, "==", Literal(0)),
                )

        elif mnemonic in cls.instrs_compare:
            inputs = [arg for arg in args if isinstance(arg, Register)]
            outputs = [Register("condition_bit")]
            eval_fn = lambda s, a: s.set_reg(
                Register("condition_bit"),
                cls.instrs_compare[mnemonic](a),
            )
        elif mnemonic in ("bf", "bf.s", "bt", "bt.s"):
            assert len(args) == 1
            inputs = [Register("condition_bit")]
            jump_target = get_jump_target(args[0])
            is_conditional = True
            has_delay_slot = mnemonic.endswith(".s")

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                condition = condition_from_expr(a.regs[Register("condition_bit")])
                if mnemonic.startswith("bf"):
                    condition = condition.negated()
                s.set_branch_condition(condition)

        elif mnemonic == "bra":
            assert len(args) == 1
            jump_target = get_jump_target(args[0])
            has_delay_slot = True
        elif mnemonic == "jmp":
            assert (
                len(args) == 1
                and isinstance(args[0], AsmAddressMode)
                and args[0].addend == AsmLiteral(0)
            )
            inputs = [args[0].base]
            jump_target = args[0].base
            is_conditional = True
            has_delay_slot = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                address = a.raw_arg(0)
                assert isinstance(address, AsmAddressMode)
                s.set_switch_expr(a.regs[address.base])

        elif mnemonic == "jsr":
            assert (
                len(args) == 1
                and isinstance(args[0], AsmAddressMode)
                and args[0].addend == AsmLiteral(0)
            )
            target_reg = args[0].base
            inputs = [*cls.argument_regs, target_reg]
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            function_target = target_reg
            has_delay_slot = True
            eval_fn = lambda s, a: s.make_function_call(a.regs[target_reg], outputs)
        elif mnemonic in ("sdiv.fictive", "udiv.fictive"):
            assert args == [Register("r4"), Register("r5")]
            inputs = [Register("r4"), Register("r5")]
            outputs = [Register("r0")]
            op = BinaryOp.sint if mnemonic == "sdiv.fictive" else BinaryOp.uint
            eval_fn = lambda s, a: s.set_reg(
                Register("r0"), op(a.reg(0), "/", a.reg(1))
            )
        elif mnemonic == "tablejmp.fictive":
            assert len(args) >= 2 and isinstance(args[0], Register)
            targets = []
            for arg in args[1:]:
                assert isinstance(arg, AsmGlobalSymbol)
                targets.append(JumpTarget(arg.symbol_name))
            inputs = [args[0]]
            jump_target = targets
            is_conditional = True
            has_delay_slot = True
            eval_fn = lambda s, a: s.set_switch_expr(a.reg(0), just_index=True)
        elif mnemonic == "movt":
            assert len(args) == 1 and isinstance(args[0], Register)
            inputs = [Register("condition_bit")]
            outputs = [args[0]]
            eval_fn = lambda s, a: s.set_reg(
                a.reg_ref(0), a.regs[Register("condition_bit")]
            )
        elif mnemonic in ("clrt", "sett"):
            assert not args
            outputs = [Register("condition_bit")]
            val = 1 if mnemonic == "sett" else 0
            eval_fn = lambda s, a: s.set_reg(Register("condition_bit"), Literal(val))
        elif mnemonic == "negatet.fictive":
            assert not args
            inputs = outputs = [Register("condition_bit")]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                condition = condition_from_expr(a.regs[Register("condition_bit")])
                s.set_reg(Register("condition_bit"), condition.negated())

        elif mnemonic == "stc":
            assert (
                len(args) == 2
                and isinstance(args[0], Register)
                and isinstance(args[1], Register)
            )
            outputs = [args[1]]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                fn_name = "M2C_LOAD_" + a.reg_ref(0).register_name.upper()
                s.set_reg(a.reg_ref(1), fn_op(fn_name, [], Type.u32()))

        elif mnemonic == "ldc":
            assert (
                len(args) == 2
                and isinstance(args[0], Register)
                and isinstance(args[1], Register)
            )
            inputs = [args[0]]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                fn_name = "M2C_STORE_" + a.reg_ref(1).register_name.upper()
                s.write_statement(void_fn_op(fn_name, [a.reg(0)]))

        else:
            instr_str = str(AsmInstruction(mnemonic, args))

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                error = ErrorExpr(f"unknown instruction: {instr_str}")
                s.write_statement(ExprStmt(error))

        return Instruction(
            mnemonic=mnemonic,
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=clobbers,
            outputs=outputs,
            eval_fn=eval_fn,
            jump_target=jump_target,
            function_target=function_target,
            is_conditional=is_conditional,
            is_return=is_return,
            is_load=is_load,
            is_store=is_store,
            has_delay_slot=has_delay_slot,
        )

    instrs_compare: InstrMap = {
        "cmp/pl": lambda a: BinaryOp.scmp(a.reg(0), ">", Literal(0)),
        "cmp/pz": lambda a: BinaryOp.scmp(a.reg(0), ">=", Literal(0)),
        "cmp/eq": lambda a: BinaryOp.icmp(a.reg(1), "==", a.reg_or_s8_imm(0)),
        "cmp/ge": lambda a: BinaryOp.scmp(a.reg(1), ">=", a.reg(0)),
        "cmp/gt": lambda a: BinaryOp.scmp(a.reg(1), ">", a.reg(0)),
        "cmp/hi": lambda a: BinaryOp.ucmp(a.reg(1), ">", a.reg(0)),
        "cmp/hs": lambda a: BinaryOp.ucmp(a.reg(1), ">=", a.reg(0)),
        "tst": lambda a: (
            BinaryOp.icmp(a.reg(1), "==", Literal(0))
            if a.reg_ref(1) == a.raw_arg(0)
            else BinaryOp.icmp(
                BinaryOp.intptr(a.reg(1), "&", a.reg_or_s8_imm(0)), "==", Literal(0)
            )
        ),
    }

    instrs_read_modify_write: InstrMap = {
        # sh2 format is src, dst
        # add handler is dest, left, right
        "add": lambda a: (
            handle_add(replace(a, raw_args=[a.raw_arg(1), a.raw_arg(1), a.raw_arg(0)]))
            if isinstance(a.raw_arg(0), Register)
            else handle_addi(a.reg_ref(1), a.reg_ref(1), a.reg(1), a.s8_imm(0), a)
        ),
        "sub": lambda a: handle_sub(a.reg(1), a.reg(0)),
        "and": lambda a: BinaryOp.int(a.reg(1), "&", a.reg_or_imm(0)),
        "or": lambda a: handle_or(a.reg(1), a.reg_or_imm(0)),
        "xor": lambda a: handle_xor(a.reg(1), a.reg_or_imm(0)),
    }

    instrs_source_dest: InstrMap = {
        "not": lambda a: handle_bitinv(a.reg(0)),
        "neg": lambda a: UnaryOp.sint("-", a.reg(0)),
        "exts.b": lambda a: as_type(a.reg(0), Type.s8(), silent=False),
        "exts.w": lambda a: as_type(a.reg(0), Type.s16(), silent=False),
        "extu.b": lambda a: as_type(a.reg(0), Type.u8(), silent=False),
        "extu.w": lambda a: as_type(a.reg(0), Type.u16(), silent=False),
        "swap.w": lambda a: BinaryOp.int(
            BinaryOp.uint(a.reg(0), "<<", Literal(16)),
            "|",
            BinaryOp.uint(a.reg(0), ">>", Literal(16)),
        ),
    }

    instrs_multiply: InstrMap = {
        "mul.l": lambda a: BinaryOp.int(a.reg(1), "*", a.reg(0)),
        "muls": lambda a: BinaryOp.sint(as_s16(a.reg(1)), "*", as_s16(a.reg(0))),
        "mulu": lambda a: BinaryOp.uint(as_u16(a.reg(1)), "*", as_u16(a.reg(0))),
    }

    instrs_shift: InstrMap = {
        "shll": lambda a: fold_mul_chains(
            BinaryOp.int(a.reg(0), "<<", Literal(1)), allow_sll_chains=True
        ),
        "shlr": lambda a: fold_shift_right(a.reg(0), 1, signed=False),
        "shar": lambda a: fold_shift_right(a.reg(0), 1, signed=True),
        "shll2": lambda a: fold_mul_chains(
            BinaryOp.int(a.reg(0), "<<", Literal(2)), allow_sll_chains=True
        ),
        "shlr2": lambda a: fold_shift_right(a.reg(0), 2, signed=False),
        "shll8": lambda a: fold_mul_chains(
            BinaryOp.int(a.reg(0), "<<", Literal(8)), allow_sll_chains=True
        ),
        "shlr8": lambda a: fold_shift_right(a.reg(0), 8, signed=False),
        "shll16": lambda a: fold_mul_chains(
            BinaryOp.int(a.reg(0), "<<", Literal(16)), allow_sll_chains=True
        ),
        "shlr16": lambda a: fold_shift_right(a.reg(0), 16, signed=False),
        "rotl": lambda a: BinaryOp.uint(
            BinaryOp.uint(a.reg(0), "<<", Literal(1)),
            "|",
            BinaryOp.uint(a.reg(0), ">>", Literal(31)),
        ),
        "rotr": lambda a: BinaryOp.uint(
            BinaryOp.uint(a.reg(0), ">>", Literal(1)),
            "|",
            BinaryOp.uint(a.reg(0), "<<", Literal(31)),
        ),
    }

    asm_patterns = [
        DivisionHelperPattern(),
        JumpTablePattern(),
        Sh2AddrModeWritebackPattern(),
        NegateTPattern(),
        SubcSelfPattern(),
    ]

    def arg_name(self, loc: ArgLoc) -> str:
        if loc.offset is not None:
            return f"arg{loc.offset // 4 + 4}"
        assert loc.reg is not None
        reg_num = int(loc.reg.register_name[1:])
        return f"arg{reg_num - 4}"

    def function_abi(
        self,
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> Abi:
        known_slots: List[AbiArgSlot] = []
        candidate_slots: List[AbiArgSlot] = []
        possible_slots: List[AbiArgSlot] = []

        if fn_sig.params_known:
            if (
                fn_sig.return_type.is_struct()
                and fn_sig.return_type.get_parameter_size_align_bytes()[0] > 8
            ):
                known_slots.append(
                    AbiArgSlot(
                        ArgLoc(None, -1, Register("r2")),
                        Type.ptr(fn_sig.return_type),
                        name="__return__",
                        comment="return",
                    )
                )

            reg_num = 0
            for param in fn_sig.params:
                param_type = param.type.decay()
                # (The sh2 ABI we target does not use alignment > 4 anywhere,
                # so we can ignore that aspect here)
                size, _ = param_type.get_parameter_size_align_bytes()
                size_regs = (size + 3) // 4
                name = param.name
                for i in range(reg_num, reg_num + size_regs):
                    unk_offset = 4 * (i - reg_num)
                    reg = Register(f"r{i + 4}") if i < 4 else None
                    stack_loc = None if reg is not None else (i - 4) * 4
                    if size_regs > 1:
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
                            ArgLoc(stack_loc, i, reg),
                            sub_type,
                            name=name2,
                            comment=comment,
                        )
                    )
                reg_num += size_regs

            if fn_sig.is_variadic:
                for i in range(reg_num, 4):
                    candidate_slots.append(
                        AbiArgSlot(
                            ArgLoc(None, i, Register(f"r{i + 4}")), Type.intptr()
                        )
                    )
        else:
            candidate_slots = [
                AbiArgSlot(ArgLoc(None, i, Register(f"r{i + 4}")), Type.intptr())
                for i in range(4)
            ]

        # Skip candidate slots starting from the first unset register, or past
        # the last likely passed register.
        highest_likely = -1
        for i in range(4, 8):
            if Register(f"r{i}") not in likely_regs:
                break
            if likely_regs[Register(f"r{i}")]:
                highest_likely = i

        for slot in candidate_slots:
            assert slot.loc.reg is not None
            if int(slot.loc.reg.register_name[1:]) <= highest_likely:
                possible_slots.append(slot)

        return Abi(
            arg_slots=known_slots,
            possible_slots=possible_slots,
        )

    @staticmethod
    def function_return(expr: Expression) -> Dict[Register, Expression]:
        return {
            Register("r0"): Cast(
                expr, reinterpret=True, silent=True, type=Type.intptr()
            ),
            Register("r1"): as_u32(
                Cast(expr, reinterpret=True, silent=False, type=Type.u64())
            ),
        }
