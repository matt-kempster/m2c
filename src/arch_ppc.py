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
    ImplicitInstrMap,
    PPCCmpInstrMap,
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
    handle_addis,
    handle_bgez,
    handle_conditional_move,
    handle_convert,
    handle_la,
    handle_load,
    handle_loadx,
    handle_rlwinm,
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
    make_storex,
    void_fn_op,
)


class PpcArch(Arch):
    stack_pointer_reg = Register("r1")
    frame_pointer_reg = None
    return_address_reg = Register("lr")

    base_return_regs = [Register(r) for r in ["r3", "f1"]]
    all_return_regs = [Register(r) for r in ["f1", "f2", "r3", "r4"]]
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
            # PPC: $r2 & $r13 are used for the small-data region, and are like $gp in MIPS
            "lr",
            # TODO: Some of the bits in CR are required to be saved (but usually the whole reg is?)
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

    instrs_ignore: InstrSet = {
        "nop",
        "j",
        # PPC: assume stmw/lmw are only used for saving/restoring saved regs
        "stmw",
        "lmw",
        # PPC: `{crclr,crset} 6` are used as part of the ABI for floats & varargs
        # For now, we can ignore them (and later use them to help in ppc_function_abi)
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
        # PPC
        # TODO: Technically `bge` is defined as `cr0_gt || cr0_eq`; not as `!cr0_lt`
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
    }
    instrs_decctr_branches: CmpInstrMap = {
        # PPC
        # Decrement the CTR register, then branch
        "bdnz": lambda a: a.cmp_reg("ctr"),
        "bdz": lambda a: a.cmp_reg("ctr").negated(),
    }
    instrs_float_branches: InstrSet = {}
    instrs_jumps: InstrSet = {
        # Unconditional jump
        # "jr",
        # PPC
        "b",
        "blr",
        "bctr",
    }
    instrs_fn_call: InstrSet = {
        # Function call
        "bl",
        "blrl",
    }
    instrs_no_dest: StmtInstrMap = {
        # Conditional traps (happen with Pascal code sometimes, might as well give a nicer
        # output than MIPS2C_ERROR(...))
        "break": lambda a: void_fn_op(
            "MIPS2C_BREAK", [a.imm(0)] if a.count() >= 1 else []
        ),
        "sync": lambda a: void_fn_op("MIPS2C_SYNC", []),
        "trapuv.fictive": lambda a: CommentStmt("code compiled with -trapuv"),
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
        "ori": lambda a: handle_or(a, a.reg(1), a.unsigned_imm(2)),
        "oris": lambda a: handle_or(a, a.reg(1), a.shifted_imm(2)),
        "and": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.reg(2)),
        "or": lambda a: BinaryOp.int(left=a.reg(1), op="|", right=a.reg(2)),
        "not": lambda a: UnaryOp("~", a.reg(1), type=Type.intish()),
        "nor": lambda a: UnaryOp(
            "~", BinaryOp.int(left=a.reg(1), op="|", right=a.reg(2)), type=Type.intish()
        ),
        "xor": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.reg(2)),
        "andi": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.unsigned_imm(2)),
        "andis": lambda a: BinaryOp.int(left=a.reg(1), op="&", right=a.shifted_imm(2)),
        "xori": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.unsigned_imm(2)),
        "xoris": lambda a: BinaryOp.int(left=a.reg(1), op="^", right=a.shifted_imm(2)),
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
        # PPC
        "add": lambda a: handle_add(a),
        "addis": lambda a: handle_addis(a),
        "subf": lambda a: fold_gcc_divmod(
            BinaryOp.intptr(left=a.reg(2), op="-", right=a.reg(1))
        ),
        "divw": lambda a: BinaryOp.s32(a.reg(1), "/", a.reg(2)),
        "divuw": lambda a: BinaryOp.u32(a.reg(1), "/", a.reg(2)),
        "mulli": lambda a: BinaryOp.int(a.reg(1), "*", a.imm(2)),
        "mullw": lambda a: BinaryOp.int(a.reg(1), "*", a.reg(2)),
        "mulhw": lambda a: fold_gcc_divmod(BinaryOp.int(a.reg(1), "MULT_HI", a.reg(2))),
        "mulhwu": lambda a: fold_gcc_divmod(
            BinaryOp.int(a.reg(1), "MULTU_HI", a.reg(2))
        ),
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
        "lis": lambda a: load_upper(a),
        "extsb": lambda a: handle_convert(a.reg(1), Type.s8(), Type.intish()),
        "extsh": lambda a: handle_convert(a.reg(1), Type.s16(), Type.intish()),
        "mflr": lambda a: a.regs[Register("lr")],
        "mfctr": lambda a: a.regs[Register("ctr")],
        "mr": lambda a: a.reg(1),
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
        "srw": lambda a: fold_gcc_divmod(
            BinaryOp(
                left=as_u32(a.reg(1)),
                op=">>",
                right=as_intish(a.reg(2)),
                type=Type.u32(),
            )
        ),
        "sraw": lambda a: fold_gcc_divmod(
            BinaryOp(
                left=as_s32(a.reg(1)),
                op=">>",
                right=as_intish(a.reg(2)),
                type=Type.s32(),
            )
        ),
        "srawi": lambda a: handle_sra(a),
        "notnot.fictive": lambda a: UnaryOp(op="!!", expr=a.reg(1), type=Type.intish()),
        # TODO: Do we need to model the promotion from f32 to f64 here?
        "lfs": lambda a: handle_load(a, type=Type.f32()),
        "lfd": lambda a: handle_load(a, type=Type.f64()),
        "lfsx": lambda a: handle_loadx(a, type=Type.f32()),
        "lfdx": lambda a: handle_loadx(a, type=Type.f64()),
        # PPC Floating Point
        "fadd": lambda a: BinaryOp.f64(a.reg(1), "+", a.reg(2)),
        "fadds": lambda a: BinaryOp.f32(a.reg(1), "+", a.reg(2)),
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
        # PPC Floating Poing Fused Multiply-{Add,Sub}
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

    # TODO: Unclear if there will be many instructions like this, or if
    # there will just be separate dicts for each implicit dest register
    instrs_implicit_destination: ImplicitInstrMap = {
        "mtlr": (Register("lr"), lambda a: a.reg(0)),
        "mtctr": (Register("ctr"), lambda a: a.reg(0)),
    }

    instrs_ppc_compare: PPCCmpInstrMap = {
        "cmpw": lambda a, op: BinaryOp.spcmp(a.reg(0), op, a.reg(1)),
        "cmpwi": lambda a, op: BinaryOp.spcmp(a.reg(0), op, a.imm(1)),
        "cmplw": lambda a, op: BinaryOp.upcmp(a.reg(0), op, a.reg(1)),
        "cmplwi": lambda a, op: BinaryOp.upcmp(a.reg(0), op, a.imm(1)),
        # TODO: There is a difference in how these two instructions handle NaN
        # TODO: Assert that the first arg is cr0
        "fcmpo": lambda a, op: BinaryOp.fcmp(a.reg(1), op, a.reg(2)),
        "fcmpu": lambda a, op: BinaryOp.fcmp(a.reg(1), op, a.reg(2)),
        "fcmpo.lte": lambda a, op: BinaryOp.fcmp(
            a.reg(1), op if op != "==" else "<=", a.reg(2)
        ),
        "fcmpo.gte": lambda a, op: BinaryOp.fcmp(
            a.reg(1), op if op != "==" else ">=", a.reg(2)
        ),
    }

    def function_abi(
        self,
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> Abi:
        known_slots: List[AbiArgSlot] = []
        candidate_slots: List[AbiArgSlot] = []

        # TODO: We don't actually know the order
        intptr_regs = [r for r in self.argument_regs if r.register_name[0] != "f"]
        float_regs = [r for r in self.argument_regs if r.register_name[0] == "f"]

        if fn_sig.params_known:
            # TODO: Parse function signatures
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
            for ind, reg in enumerate(self.argument_regs):
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
                # PPC is simple; only r3-r10/f1-f13 can be used for arguments
                regname = slot.reg.register_name
                prev_reg = Register(f"{regname[0]}{int(regname[1:])-1}")
                if prev_reg in self.argument_regs and prev_reg not in valid_extra_regs:
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

    def function_return(self, expr: Expression) -> List[Tuple[Register, Expression]]:
        return [
            (
                Register("f1"),
                Cast(expr, reinterpret=True, silent=True, type=Type.floatish()),
            ),
            (
                Register("r3"),
                Cast(expr, reinterpret=True, silent=True, type=Type.intptr()),
            ),
            (
                Register("r4"),
                as_u32(Cast(expr, reinterpret=True, silent=False, type=Type.u64())),
            ),
            (Register("f2"), SecondF64Half()),
        ]
