from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from .types import FunctionSignature, Type
from .parse_instruction import Register
from .translate import (
    Abi,
    AbiArgSlot,
    Arch,
    BinaryOp,
    Cast,
    CmpInstrMap,
    CommentStmt,
    ErrorExpr,
    Expression,
    InstrMap,
    InstrSet,
    Literal,
    PairInstrMap,
    SecondF64Half,
    StmtInstrMap,
    StoreInstrMap,
    UnaryOp,
    as_f32,
    as_f64,
    as_int64,
    as_intish,
    as_intptr,
    as_ptr,
    as_s32,
    as_s64,
    as_type,
    as_u32,
    as_u64,
    fn_op,
    fold_gcc_divmod,
    fold_mul_chains,
    handle_add,
    handle_add_double,
    handle_add_float,
    handle_addi,
    handle_bgez,
    handle_conditional_move,
    handle_convert,
    handle_la,
    handle_load,
    handle_lwl,
    handle_lwr,
    handle_ori,
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


class MipsArch(Arch):
    stack_pointer_reg = Register("sp")
    frame_pointer_reg = Register("fp")
    return_address_reg = Register("ra")

    base_return_regs = [Register(r) for r in ["v0", "f0"]]
    all_return_regs = [Register(r) for r in ["v0", "v1", "f0", "f1"]]
    argument_regs = [Register(r) for r in ["a0", "a1", "a2", "a3", "f12", "f14"]]
    simple_temp_regs = [
        Register(r)
        for r in [
            "v0",
            "v1",
            "t0",
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "t7",
            "t8",
            "t9",
            "f0",
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
            "f13",
            "f15",
            "f16",
            "f17",
            "f18",
            "f19",
        ]
    ]
    temp_regs = (
        argument_regs
        + simple_temp_regs
        + [
            Register(r)
            for r in [
                "at",
                "hi",
                "lo",
                "condition_bit",
            ]
        ]
    )
    saved_regs = [
        Register(r)
        for r in [
            "s0",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
            "s7",
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
            "ra",
            "31",
            "fp",
            "gp",
        ]
    ]

    instrs_ignore: InstrSet = {
        # Ignore FCSR sets; they are leftovers from float->unsigned conversions.
        # FCSR gets are as well, but it's fine to read MIPS2C_ERROR for those.
        "ctc1",
        "nop",
        "b",
        "j",
    }
    instrs_store: StoreInstrMap = {
        # Storage instructions
        "sb": lambda a: make_store(a, type=Type.int_of_size(8)),
        "sh": lambda a: make_store(a, type=Type.int_of_size(16)),
        "sw": lambda a: make_store(a, type=Type.reg32(likely_float=False)),
        "sd": lambda a: make_store(a, type=Type.reg64(likely_float=False)),
        # Unaligned stores
        "swl": lambda a: handle_swl(a),
        "swr": lambda a: handle_swr(a),
        # Floating point storage/conversion
        "swc1": lambda a: make_store(a, type=Type.reg32(likely_float=True)),
        "sdc1": lambda a: make_store(a, type=Type.reg64(likely_float=True)),
    }
    instrs_branches: CmpInstrMap = {
        # Branch instructions/pseudoinstructions
        "beq": lambda a: BinaryOp.icmp(a.reg(0), "==", a.reg(1)),
        "bne": lambda a: BinaryOp.icmp(a.reg(0), "!=", a.reg(1)),
        "beqz": lambda a: BinaryOp.icmp(a.reg(0), "==", Literal(0)),
        "bnez": lambda a: BinaryOp.icmp(a.reg(0), "!=", Literal(0)),
        "blez": lambda a: BinaryOp.scmp(a.reg(0), "<=", Literal(0)),
        "bgtz": lambda a: BinaryOp.scmp(a.reg(0), ">", Literal(0)),
        "bltz": lambda a: BinaryOp.scmp(a.reg(0), "<", Literal(0)),
        "bgez": lambda a: handle_bgez(a),
    }
    instrs_float_branches: InstrSet = {
        # Floating-point branch instructions
        "bc1t",
        "bc1f",
    }
    instrs_jumps: InstrSet = {
        # Unconditional jump
        "jr"
    }
    instrs_fn_call: InstrSet = {
        # Function call
        "jal",
        "jalr",
    }
    instrs_no_dest: StmtInstrMap = {
        # Conditional traps (happen with Pascal code sometimes, might as well give a nicer
        # output than MIPS2C_ERROR(...))
        "teq": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.icmp(a.reg(0), "==", a.reg(1))]
        ),
        "tne": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.icmp(a.reg(0), "!=", a.reg(1))]
        ),
        "tlt": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.scmp(a.reg(0), "<", a.reg(1))]
        ),
        "tltu": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.ucmp(a.reg(0), "<", a.reg(1))]
        ),
        "tge": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.scmp(a.reg(0), ">=", a.reg(1))]
        ),
        "tgeu": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.ucmp(a.reg(0), ">=", a.reg(1))]
        ),
        "teqi": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.icmp(a.reg(0), "==", a.imm(1))]
        ),
        "tnei": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.icmp(a.reg(0), "!=", a.imm(1))]
        ),
        "tlti": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.scmp(a.reg(0), "<", a.imm(1))]
        ),
        "tltiu": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.ucmp(a.reg(0), "<", a.imm(1))]
        ),
        "tgei": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.scmp(a.reg(0), ">=", a.imm(1))]
        ),
        "tgeiu": lambda a: void_fn_op(
            "MIPS2C_TRAP_IF", [BinaryOp.ucmp(a.reg(0), ">=", a.imm(1))]
        ),
        "break": lambda a: void_fn_op(
            "MIPS2C_BREAK", [a.imm(0)] if a.count() >= 1 else []
        ),
        "sync": lambda a: void_fn_op("MIPS2C_SYNC", []),
        "trapuv.fictive": lambda a: CommentStmt("code compiled with -trapuv"),
    }
    instrs_float_comp: CmpInstrMap = {
        # Float comparisons that don't raise exception on nan
        "c.eq.s": lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)),
        "c.olt.s": lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1)),
        "c.oge.s": lambda a: BinaryOp.fcmp(a.reg(0), ">=", a.reg(1)),
        "c.ole.s": lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1)),
        "c.ogt.s": lambda a: BinaryOp.fcmp(a.reg(0), ">", a.reg(1)),
        "c.neq.s": lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)).negated(),
        "c.uge.s": lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1)).negated(),
        "c.ult.s": lambda a: BinaryOp.fcmp(a.reg(0), ">=", a.reg(1)).negated(),
        "c.ugt.s": lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1)).negated(),
        "c.ule.s": lambda a: BinaryOp.fcmp(a.reg(0), ">", a.reg(1)).negated(),
        # Float comparisons that may raise exception on nan
        "c.seq.s": lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)),
        "c.lt.s": lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1)),
        "c.ge.s": lambda a: BinaryOp.fcmp(a.reg(0), ">=", a.reg(1)),
        "c.le.s": lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1)),
        "c.gt.s": lambda a: BinaryOp.fcmp(a.reg(0), ">", a.reg(1)),
        "c.sne.s": lambda a: BinaryOp.fcmp(a.reg(0), "==", a.reg(1)).negated(),
        "c.nle.s": lambda a: BinaryOp.fcmp(a.reg(0), "<=", a.reg(1)).negated(),
        "c.nlt.s": lambda a: BinaryOp.fcmp(a.reg(0), "<", a.reg(1)).negated(),
        "c.nge.s": lambda a: BinaryOp.fcmp(a.reg(0), ">=", a.reg(1)).negated(),
        "c.ngt.s": lambda a: BinaryOp.fcmp(a.reg(0), ">", a.reg(1)).negated(),
        # Double comparisons that don't raise exception on nan
        "c.eq.d": lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)),
        "c.olt.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1)),
        "c.oge.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">=", a.dreg(1)),
        "c.ole.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1)),
        "c.ogt.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">", a.dreg(1)),
        "c.neq.d": lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)).negated(),
        "c.uge.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1)).negated(),
        "c.ult.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">=", a.dreg(1)).negated(),
        "c.ugt.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1)).negated(),
        "c.ule.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">", a.dreg(1)).negated(),
        # Double comparisons that may raise exception on nan
        "c.seq.d": lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)),
        "c.lt.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1)),
        "c.ge.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">=", a.dreg(1)),
        "c.le.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1)),
        "c.gt.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">", a.dreg(1)),
        "c.sne.d": lambda a: BinaryOp.dcmp(a.dreg(0), "==", a.dreg(1)).negated(),
        "c.nle.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<=", a.dreg(1)).negated(),
        "c.nlt.d": lambda a: BinaryOp.dcmp(a.dreg(0), "<", a.dreg(1)).negated(),
        "c.nge.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">=", a.dreg(1)).negated(),
        "c.ngt.d": lambda a: BinaryOp.dcmp(a.dreg(0), ">", a.dreg(1)).negated(),
    }
    instrs_hi_lo: PairInstrMap = {
        # Div and mul output two results, to LO/HI registers. (Format: (hi, lo))
        "div": lambda a: (
            BinaryOp.s32(a.reg(0), "%", a.reg(1)),
            BinaryOp.s32(a.reg(0), "/", a.reg(1)),
        ),
        "divu": lambda a: (
            BinaryOp.u32(a.reg(0), "%", a.reg(1)),
            BinaryOp.u32(a.reg(0), "/", a.reg(1)),
        ),
        "ddiv": lambda a: (
            BinaryOp.s64(a.reg(0), "%", a.reg(1)),
            BinaryOp.s64(a.reg(0), "/", a.reg(1)),
        ),
        "ddivu": lambda a: (
            BinaryOp.u64(a.reg(0), "%", a.reg(1)),
            BinaryOp.u64(a.reg(0), "/", a.reg(1)),
        ),
        "mult": lambda a: (
            fold_gcc_divmod(BinaryOp.int(a.reg(0), "MULT_HI", a.reg(1))),
            BinaryOp.int(a.reg(0), "*", a.reg(1)),
        ),
        "multu": lambda a: (
            fold_gcc_divmod(BinaryOp.int(a.reg(0), "MULTU_HI", a.reg(1))),
            BinaryOp.int(a.reg(0), "*", a.reg(1)),
        ),
        "dmult": lambda a: (
            BinaryOp.int64(a.reg(0), "DMULT_HI", a.reg(1)),
            BinaryOp.int64(a.reg(0), "*", a.reg(1)),
        ),
        "dmultu": lambda a: (
            BinaryOp.int64(a.reg(0), "DMULTU_HI", a.reg(1)),
            BinaryOp.int64(a.reg(0), "*", a.reg(1)),
        ),
    }
    instrs_source_first: InstrMap = {
        # Floating point moving instruction
        "mtc1": lambda a: a.reg(0)
    }
    instrs_destination_first: InstrMap = {
        # Flag-setting instructions
        "slt": lambda a: BinaryOp.scmp(a.reg(1), "<", a.reg(2)),
        "slti": lambda a: BinaryOp.scmp(a.reg(1), "<", a.imm(2)),
        "sltu": lambda a: handle_sltu(a),
        "sltiu": lambda a: handle_sltiu(a),
        # Integer arithmetic
        "addi": lambda a: handle_addi(a),
        "addiu": lambda a: handle_addi(a),
        "addu": lambda a: handle_add(a),
        "subu": lambda a: (
            fold_mul_chains(fold_gcc_divmod(BinaryOp.intptr(a.reg(1), "-", a.reg(2))))
        ),
        "negu": lambda a: fold_mul_chains(
            UnaryOp(op="-", expr=as_s32(a.reg(1)), type=Type.s32())
        ),
        "neg": lambda a: fold_mul_chains(
            UnaryOp(op="-", expr=as_s32(a.reg(1)), type=Type.s32())
        ),
        "div.fictive": lambda a: BinaryOp.s32(a.reg(1), "/", a.full_imm(2)),
        "mod.fictive": lambda a: BinaryOp.s32(a.reg(1), "%", a.full_imm(2)),
        # 64-bit integer arithmetic, treated mostly the same as 32-bit for now
        "daddi": lambda a: handle_addi(a),
        "daddiu": lambda a: handle_addi(a),
        "daddu": lambda a: handle_add(a),
        "dsubu": lambda a: fold_mul_chains(BinaryOp.intptr(a.reg(1), "-", a.reg(2))),
        "dnegu": lambda a: fold_mul_chains(
            UnaryOp(op="-", expr=as_s64(a.reg(1)), type=Type.s64())
        ),
        "dneg": lambda a: fold_mul_chains(
            UnaryOp(op="-", expr=as_s64(a.reg(1)), type=Type.s64())
        ),
        # Hi/lo register uses (used after division/multiplication)
        "mfhi": lambda a: a.regs[Register("hi")],
        "mflo": lambda a: a.regs[Register("lo")],
        # Floating point arithmetic
        "add.s": lambda a: handle_add_float(a),
        "sub.s": lambda a: BinaryOp.f32(a.reg(1), "-", a.reg(2)),
        "neg.s": lambda a: UnaryOp("-", as_f32(a.reg(1)), type=Type.f32()),
        "abs.s": lambda a: fn_op("fabsf", [as_f32(a.reg(1))], Type.f32()),
        "sqrt.s": lambda a: fn_op("sqrtf", [as_f32(a.reg(1))], Type.f32()),
        "div.s": lambda a: BinaryOp.f32(a.reg(1), "/", a.reg(2)),
        "mul.s": lambda a: BinaryOp.f32(a.reg(1), "*", a.reg(2)),
        # Double-precision arithmetic
        "add.d": lambda a: handle_add_double(a),
        "sub.d": lambda a: BinaryOp.f64(a.dreg(1), "-", a.dreg(2)),
        "neg.d": lambda a: UnaryOp("-", as_f64(a.dreg(1)), type=Type.f64()),
        "abs.d": lambda a: fn_op("fabs", [as_f64(a.dreg(1))], Type.f64()),
        "sqrt.d": lambda a: fn_op("sqrt", [as_f64(a.dreg(1))], Type.f64()),
        "div.d": lambda a: BinaryOp.f64(a.dreg(1), "/", a.dreg(2)),
        "mul.d": lambda a: BinaryOp.f64(a.dreg(1), "*", a.dreg(2)),
        # Floating point conversions
        "cvt.d.s": lambda a: handle_convert(a.reg(1), Type.f64(), Type.f32()),
        "cvt.d.w": lambda a: handle_convert(a.reg(1), Type.f64(), Type.intish()),
        "cvt.s.d": lambda a: handle_convert(a.dreg(1), Type.f32(), Type.f64()),
        "cvt.s.w": lambda a: handle_convert(a.reg(1), Type.f32(), Type.intish()),
        "cvt.w.d": lambda a: handle_convert(a.dreg(1), Type.s32(), Type.f64()),
        "cvt.w.s": lambda a: handle_convert(a.reg(1), Type.s32(), Type.f32()),
        "cvt.s.u.fictive": lambda a: handle_convert(a.reg(1), Type.f32(), Type.u32()),
        "cvt.u.d.fictive": lambda a: handle_convert(a.dreg(1), Type.u32(), Type.f64()),
        "cvt.u.s.fictive": lambda a: handle_convert(a.reg(1), Type.u32(), Type.f32()),
        "trunc.w.s": lambda a: handle_convert(a.reg(1), Type.s32(), Type.f32()),
        "trunc.w.d": lambda a: handle_convert(a.dreg(1), Type.s32(), Type.f64()),
        # Bit arithmetic
        "ori": lambda a: handle_ori(a),
        "and": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.reg(2)),
        "or": lambda a: BinaryOp.int(left=a.reg(1), op="|", right=a.reg(2)),
        "not": lambda a: UnaryOp("~", a.reg(1), type=Type.intish()),
        "nor": lambda a: UnaryOp(
            "~", BinaryOp.int(left=a.reg(1), op="|", right=a.reg(2)), type=Type.intish()
        ),
        "xor": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.reg(2)),
        "andi": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.unsigned_imm(2)),
        "xori": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.unsigned_imm(2)),
        # Shifts
        "sll": lambda a: fold_mul_chains(
            BinaryOp.int(left=a.reg(1), op="<<", right=as_intish(a.imm(2)))
        ),
        "sllv": lambda a: fold_mul_chains(
            BinaryOp.int(left=a.reg(1), op="<<", right=as_intish(a.reg(2)))
        ),
        "srl": lambda a: fold_gcc_divmod(
            BinaryOp(
                left=as_u32(a.reg(1)),
                op=">>",
                right=as_intish(a.imm(2)),
                type=Type.u32(),
            )
        ),
        "srlv": lambda a: fold_gcc_divmod(
            BinaryOp(
                left=as_u32(a.reg(1)),
                op=">>",
                right=as_intish(a.reg(2)),
                type=Type.u32(),
            )
        ),
        "sra": lambda a: handle_sra(a),
        "srav": lambda a: fold_gcc_divmod(
            BinaryOp(
                left=as_s32(a.reg(1)),
                op=">>",
                right=as_intish(a.reg(2)),
                type=Type.s32(),
            )
        ),
        # 64-bit shifts
        "dsll": lambda a: fold_mul_chains(
            BinaryOp.int64(left=a.reg(1), op="<<", right=as_intish(a.imm(2)))
        ),
        "dsll32": lambda a: fold_mul_chains(
            BinaryOp.int64(left=a.reg(1), op="<<", right=imm_add_32(a.imm(2)))
        ),
        "dsllv": lambda a: BinaryOp.int64(
            left=a.reg(1), op="<<", right=as_intish(a.reg(2))
        ),
        "dsrl": lambda a: BinaryOp(
            left=as_u64(a.reg(1)), op=">>", right=as_intish(a.imm(2)), type=Type.u64()
        ),
        "dsrl32": lambda a: BinaryOp(
            left=as_u64(a.reg(1)), op=">>", right=imm_add_32(a.imm(2)), type=Type.u64()
        ),
        "dsrlv": lambda a: BinaryOp(
            left=as_u64(a.reg(1)), op=">>", right=as_intish(a.reg(2)), type=Type.u64()
        ),
        "dsra": lambda a: BinaryOp(
            left=as_s64(a.reg(1)), op=">>", right=as_intish(a.imm(2)), type=Type.s64()
        ),
        "dsra32": lambda a: BinaryOp(
            left=as_s64(a.reg(1)), op=">>", right=imm_add_32(a.imm(2)), type=Type.s64()
        ),
        "dsrav": lambda a: BinaryOp(
            left=as_s64(a.reg(1)), op=">>", right=as_intish(a.reg(2)), type=Type.s64()
        ),
        # Move pseudoinstruction
        "move": lambda a: a.reg(1),
        # Floating point moving instructions
        "mfc1": lambda a: a.reg(1),
        "mov.s": lambda a: a.reg(1),
        "mov.d": lambda a: as_f64(a.dreg(1)),
        # Conditional moves
        "movn": lambda a: handle_conditional_move(a, True),
        "movz": lambda a: handle_conditional_move(a, False),
        # FCSR get
        "cfc1": lambda a: ErrorExpr("cfc1"),
        # Immediates
        "li": lambda a: a.full_imm(1),
        "lui": lambda a: load_upper(a),
        "la": lambda a: handle_la(a),
        # Loading instructions
        "lb": lambda a: handle_load(a, type=Type.s8()),
        "lbu": lambda a: handle_load(a, type=Type.u8()),
        "lh": lambda a: handle_load(a, type=Type.s16()),
        "lhu": lambda a: handle_load(a, type=Type.u16()),
        "lw": lambda a: handle_load(a, type=Type.reg32(likely_float=False)),
        "ld": lambda a: handle_load(a, type=Type.reg64(likely_float=False)),
        "lwu": lambda a: handle_load(a, type=Type.u32()),
        "lwc1": lambda a: handle_load(a, type=Type.reg32(likely_float=True)),
        "ldc1": lambda a: handle_load(a, type=Type.reg64(likely_float=True)),
        # Unaligned loads
        "lwl": lambda a: handle_lwl(a),
        "lwr": lambda a: handle_lwr(a),
    }

    def function_abi(
        self,
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> Abi:
        """Compute stack positions/registers used by a function according to the o32 ABI,
        based on C type information. Additionally computes a list of registers that might
        contain arguments, if the function is a varargs function. (Additional varargs
        arguments may be passed on the stack; we could compute the offset at which that
        would start but right now don't care -- we just slurp up everything.)"""

        known_slots: List[AbiArgSlot] = []
        candidate_slots: List[AbiArgSlot] = []
        if fn_sig.params_known:
            offset = 0
            only_floats = True
            if fn_sig.return_type.is_struct():
                # The ABI for struct returns is to pass a pointer to where it should be written
                # as the first argument.
                known_slots.append(
                    AbiArgSlot(
                        offset=0,
                        reg=Register("a0"),
                        name="__return__",
                        type=Type.ptr(fn_sig.return_type),
                    )
                )
                offset = 4
                only_floats = False

            for ind, param in enumerate(fn_sig.params):
                # Array parameters decay into pointers
                param_type = param.type.decay()
                size, align = param_type.get_parameter_size_align_bytes()
                size = (size + 3) & ~3
                only_floats = only_floats and param_type.is_float()
                offset = (offset + align - 1) & -align
                name = param.name
                reg2: Optional[Register]
                if ind < 2 and only_floats:
                    reg = Register("f12" if ind == 0 else "f14")
                    is_double = (
                        param_type.is_float() and param_type.get_size_bits() == 64
                    )
                    known_slots.append(
                        AbiArgSlot(offset=offset, reg=reg, name=name, type=param_type)
                    )
                    if is_double and not for_call:
                        name2 = f"{name}_lo" if name else None
                        reg2 = Register("f13" if ind == 0 else "f15")
                        known_slots.append(
                            AbiArgSlot(
                                offset=offset + 4,
                                reg=reg2,
                                name=name2,
                                type=Type.any_reg(),
                            )
                        )
                else:
                    for i in range(offset // 4, (offset + size) // 4):
                        unk_offset = 4 * i - offset
                        name2 = (
                            f"{name}_unk{unk_offset:X}" if name and unk_offset else name
                        )
                        reg2 = Register(f"a{i}") if i < 4 else None
                        known_slots.append(
                            AbiArgSlot(
                                offset=4 * i, reg=reg2, name=name2, type=param_type
                            )
                        )
                offset += size

            if fn_sig.is_variadic:
                for i in range(offset // 4, 4):
                    candidate_slots.append(
                        AbiArgSlot(i * 4, Register(f"a{i}"), Type.any_reg())
                    )

        else:
            candidate_slots = [
                AbiArgSlot(0, Register("f12"), Type.floatish()),
                AbiArgSlot(4, Register("f13"), Type.floatish()),
                AbiArgSlot(4, Register("f14"), Type.floatish()),
                AbiArgSlot(0, Register("a0"), Type.intptr()),
                AbiArgSlot(4, Register("a1"), Type.any_reg()),
                AbiArgSlot(8, Register("a2"), Type.any_reg()),
                AbiArgSlot(12, Register("a3"), Type.any_reg()),
            ]

        valid_extra_regs: Set[Register] = {
            slot.reg for slot in known_slots if slot.reg is not None
        }
        possible_slots: List[AbiArgSlot] = []
        for slot in candidate_slots:
            if slot.reg is None or slot.reg not in likely_regs:
                continue

            # Don't pass this register if lower numbered ones are undefined.
            # Following the o32 ABI, register order can be a prefix of either:
            # a0, a1, a2, a3
            # f12, a1, a2, a3
            # f12, f14, a2, a3
            # f12, f13, a2, a3
            # f12, f13, f14, f15
            require: Optional[List[str]] = None
            if slot == candidate_slots[0]:
                # For varargs, a subset of a0 .. a3 may be used. Don't check
                # earlier registers for the first member of that subset.
                pass
            elif slot.reg == Register("f13") or slot.reg == Register("f14"):
                require = ["f12"]
            elif slot.reg == Register("a1"):
                require = ["a0", "f12"]
            elif slot.reg == Register("a2"):
                require = ["a1", "f13", "f14"]
            elif slot.reg == Register("a3"):
                require = ["a2"]
            if require and not any(Register(r) in valid_extra_regs for r in require):
                continue

            valid_extra_regs.add(slot.reg)

            if slot.reg == Register("f13"):
                # We don't pass in f13 or f15 because they will often only
                # contain SecondF64Half(), and otherwise would need to be
                # merged with f12/f14 which we don't have logic for right
                # now. However, f13 can still matter for whether a2 should
                # be passed, and so is kept in possible_regs.
                continue

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

    def function_return(self, expr: Expression) -> List[Tuple[Register, Expression]]:
        return [
            (
                Register("f0"),
                Cast(expr, reinterpret=True, silent=True, type=Type.floatish()),
            ),
            (
                Register("v0"),
                Cast(expr, reinterpret=True, silent=True, type=Type.intptr()),
            ),
            (
                Register("v1"),
                as_u32(Cast(expr, reinterpret=True, silent=False, type=Type.u64())),
            ),
            (Register("f1"), SecondF64Half()),
        ]
