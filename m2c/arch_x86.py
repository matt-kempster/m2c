"""i386 (x86) architecture support for Intel-syntax asm.

The module provides registration, parsing, and structural instruction
information (inputs/outputs/jump targets), plus semantics (eval_fns) for the
data operations, flags, and conditionals: mov/movsx/movzx/lea/xchg, the ALU
(incl. mul/imul/div/idiv/cdq and shifts), cmp/test, all jcc and setcc, and
loads/stores through all addressing modes. x86's moving stack is handled by
an ESP-delta prepass (X86RewritePattern / rewrite_stack_ops) that
computes esp's offset from function entry at every instruction and rewrites
push/pop/call-argument/ebp-frame accesses into fixed frame offsets, so the
rest of m2c (which assumes a constant post-prologue stack pointer) works
unchanged. This recovers call arguments (cdecl and stdcall), tail calls, and
jump-table switches, plus rep string ops, loop, and rdtsc. x87 FPU support
lives in the same whole-body prepass (with its implementation in m2c/x86_fpu.py)
that eliminates the FPU register stack into flat virtual registers f0..f7,
with the per-instruction semantics in X86Arch._parse_fpu (float arithmetic/
compares/conversions, the fnstsw/test-ah compare idiom, and the float call
ABI: returns, per-callee stack deltas, and float arguments).

Design notes:

- Operand widths ("byte ptr" prefixes, sub-register names like al/ax) are
  canonicalized into ARM-style mnemonic suffixes (`mov.b`, `mov.w`, `mov.q`;
  32-bit is bare); see preprocess_instruction and normalize_instruction.

- Low-byte/word writes are modeled as writing the full storage register with
  a partial-width-typed value. Reads of ah/bh/ch/dh are lowered to explicit
  shifts from the full register; high-byte writes are rejected because they
  would require preserving both the low byte and upper 16 bits.

- Flags mirror ARM's condition flag scheme (z, n, c, v plus the composite
  hi/ge/gt pseudo-registers), except that after `cmp a, b` (or sub/neg)
  x86's carry flag is a *borrow*, the inverse of ARM's carry; see
  eval_x86_cmp and `condition_flags`.

- Arguments: cdecl passes all arguments on the stack. At function entry
  [esp + 0] holds the return address, so with an unmoved stack pointer the
  arguments live at [esp + 4], [esp + 8], ... A literal `sub esp, N` in the
  prologue is folded into StackInfo.allocated_stack_size (see
  get_stack_info), moving the argument region to [esp + N + 4]. Mid-function
  esp movement (push/pop in particular) never reaches this static view: the
  ESP-delta prepass has already rewritten it into fixed frame offsets.
"""

from __future__ import annotations
import re
import struct
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

from .error import DecompFailure
from .options import Target
from .asm_file import (
    AsmData,
    AsmFile,
    Label,
)
from .c_types import CType, TypeMap, parse_struct_member, resolve_typedefs
from m2c_pycparser import c_ast as ca
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
    ZERO,
    get_jump_target,
    traverse_arg,
)
from .asm_pattern import (
    AsmMatch,
    AsmMatcher,
    AsmPattern,
    BodyPart,
    Replacement,
    ReplacementPart,
    SimpleAsmPattern,
    make_pattern,
)
from .instruction import (
    ArchAsm,
    Instruction,
    InstructionMeta,
    Location,
    StackLocation,
)
from .ir_pattern import IrMatch, IrPattern
from .flow_graph import ArchFlowGraph, FlowGraph
from .translate import (
    Abi,
    AbiArgSlot,
    AddressMode,
    AddressOf,
    ArgLoc,
    Arch,
    ArrayAccess,
    BinaryOp,
    Cast,
    Condition,
    EvalOnceExpr,
    Expression,
    FuncCall,
    GlobalSymbol,
    InstrArgs,
    Literal,
    NodeState,
    RawSymbolRef,
    RegExpression,
    StoreStmt,
    StructAccess,
    Type,
    UnaryOp,
    as_intish,
    as_type,
    early_unwrap,
    parse_symbol_ref,
    uses_expr,
)
from .evaluate import (
    condition_from_expr,
    deref,
    fn_op,
    fold_divmod,
    fold_mul_chains,
    handle_add_real,
    handle_addi_real,
    handle_bitinv,
    handle_cmpnez,
    handle_convert,
    handle_or,
    handle_sub,
    make_store_real,
    load_rodata_constant,
    replace_bitand,
    shift_right_expr,
    split_imm_addend,
    void_fn_op,
)
from .types import FunctionSignature
from .x86_fpu import FPU_WIDTHED_MEMORY, rewrite_fpu_stack
from .x86_utils import (
    WIDTH_SUFFIXES,
    call_target_symbol,
    split_width_suffix,
    switch_jump_table_labels,
)


EAX = Register("eax")
ECX = Register("ecx")
EDX = Register("edx")
EBX = Register("ebx")
ESP = Register("esp")
EBP = Register("ebp")
ESI = Register("esi")
EDI = Register("edi")
EIP = Register("eip")
HI8A = Register.fictive("hi8a")
HI8B = Register.fictive("hi8b")

# Sub-register name -> (full register, width in bytes)
SUB_REGS: Dict[Register, Tuple[Register, int]] = {
    Register("al"): (EAX, 1),
    Register("ax"): (EAX, 2),
    Register("bl"): (EBX, 1),
    Register("bx"): (EBX, 2),
    Register("cl"): (ECX, 1),
    Register("cx"): (ECX, 2),
    Register("dl"): (EDX, 1),
    Register("dx"): (EDX, 2),
    Register("si"): (ESI, 2),
    Register("di"): (EDI, 2),
    Register("bp"): (EBP, 2),
    Register("sp"): (ESP, 2),
}

HIGH_BYTE_REGS: Dict[Register, Register] = {
    Register("ah"): EAX,
    Register("bh"): EBX,
    Register("ch"): ECX,
    Register("dh"): EDX,
}

PTR_WIDTHS: Dict[str, int] = {"byte": 1, "word": 2, "dword": 4, "qword": 8}

RE_PTR = re.compile(r"\b(byte|word|dword|qword)\s+ptr\s+", re.IGNORECASE)
RE_OFFSET = re.compile(r"\boffset\s+", re.IGNORECASE)
# Branch-distance operand hints (IDA-style: `jmp short loc_1`,
# `call near ptr foo`). Pure syntax; the target that follows is all we need.
RE_DISTANCE = re.compile(r"\b(short|near\s+ptr|far\s+ptr)\s+", re.IGNORECASE)
RE_ST_REG = re.compile(r"\bst\((\d)\)", re.IGNORECASE)
RE_SEGMENT = re.compile(r"\b([cdefgs]s):", re.IGNORECASE)

# x86 string instructions, whose operands are implicit (esi/edi/eax/ecx).
STRING_OP_MNEMONICS = {
    f"{op}{width}" for op in ("movs", "stos", "scas", "lods", "cmps") for width in "bwd"
}


def width_type(width: int) -> Type:
    """The Type used for loads/stores/casts of a given operand width.
    Sub-32-bit widths use sign-ambiguous types so that unification with
    surrounding code can decide signedness."""
    if width == 4:
        return Type.reg32(likely_float=False)
    return Type.int_of_size(width * 8)


def sign_extended_imm(value: int, width: int) -> Literal:
    """Interpret an immediate as a signed value of the given operand width
    (x86 immediates for arithmetic/compares are sign-extended)."""
    bits = width * 8
    value &= (1 << bits) - 1
    if value >= 1 << (bits - 1):
        value -= 1 << bits
    return Literal(value)


def address_expr(arg: Argument, a: InstrArgs) -> Expression:
    """Convert (part of) an Intel address-mode addend into an Expression."""
    if isinstance(arg, Register):
        return a.regs[arg]
    if isinstance(arg, AsmLiteral):
        return Literal(arg.value)
    if isinstance(arg, AsmGlobalSymbol):
        return a.stack_info.global_info.address_of_gsym(arg.symbol_name)
    if isinstance(arg, BinOp) and arg.op in ("+", "-", "*", "<<"):
        lhs = address_expr(arg.lhs, a)
        rhs = address_expr(arg.rhs, a)
        if arg.op in ("+", "-"):
            return BinaryOp.intptr(lhs, arg.op, rhs)
        return BinaryOp.int(lhs, arg.op, rhs)
    raise DecompFailure(f"Unsupported x86 address expression: {arg}")


def mem_target(
    a: InstrArgs, index: int
) -> Union[AddressMode, RawSymbolRef, Expression]:
    """Compute the target of a memory operand, as either an AddressMode
    (base register + literal offset, which also handles esp-relative stack
    accesses), a RawSymbolRef (absolute [symbol + offset]), or a generic
    address Expression (scaled-index modes, which deref turns into array
    accesses)."""
    arg = a.raw_arg(index)
    if not isinstance(arg, AsmAddressMode):
        # IDA writes a direct memory operand without brackets when it is a
        # bare symbol (`fld _FastAtanTable+0x4004`, `fadd _real`). This function
        # is only ever called on known memory operands, so such an operand is
        # an absolute [symbol (+ offset)] access.
        ref = parse_symbol_ref(arg)
        if ref is not None:
            return ref
        raise DecompFailure(f"expected a memory operand, found {arg}")
    if isinstance(arg.addend, AsmLiteral) or (
        arg.base == ZERO and parse_symbol_ref(arg.addend) is not None
    ):
        return a.memory_ref(index)
    addend = address_expr(arg.addend, a)
    if arg.base == ZERO:
        return addend
    return BinaryOp.intptr(a.regs[arg.base], "+", addend)


def mem_load(a: InstrArgs, index: int, type: Type) -> Expression:
    size = type.get_size_bytes()
    assert size is not None
    target = mem_target(a, index)
    expr = deref(target, a.regs, a.stack_info, size=size)
    const = load_rodata_constant(a, expr, type, raw_index=index, fixed=True)
    if const is not None:
        return const
    return as_type(expr, type, silent=True)


def mem_store(
    a: InstrArgs,
    index: int,
    value: Expression,
    value_reg: Optional[Register],
    type: Type,
) -> Optional[StoreStmt]:
    target = mem_target(a, index)
    source_raw: Optional[RegExpression] = None
    if value_reg is not None:
        source_raw = a.regs.get_raw(value_reg)
    return make_store_real(value, source_raw, target, a.regs, a.stack_info, type)


def op_value(
    a: InstrArgs,
    index: int,
    width: int,
    *,
    type: Optional[Type] = None,
    sign_extend_imm: bool = True,
) -> Expression:
    """Read an operand's value (register, memory, or immediate/symbol)."""
    arg = a.raw_arg(index)
    if isinstance(arg, Register):
        val = a.regs[arg]
        if width < 4 and val.type.get_size_bytes() != width:
            # Reading a sub-register: reinterpret the low bits. (If the value
            # is already exactly this wide -- e.g. it came from a same-width
            # memory load -- there is nothing to truncate.)
            val = as_type(val, type or width_type(width), silent=True, unify=False)
        return val
    if isinstance(arg, AsmAddressMode):
        return mem_load(a, index, type or width_type(width))
    imm = a.full_imm(index)
    if isinstance(imm, Literal) and sign_extend_imm:
        return sign_extended_imm(imm.value, width)
    return imm


def _is_zero_value(expr: Optional[Expression]) -> bool:
    """Whether a register's current value is a literal 0 (e.g. from a preceding
    `xor reg, reg`), used to recognize x86 zero-extend idioms."""
    if expr is None:
        return False
    uw = early_unwrap(expr)
    return isinstance(uw, Literal) and uw.value == 0


def sub_expr(lhs: Expression, rhs: Expression) -> Expression:
    val = handle_sub(lhs, rhs)
    if isinstance(val, BinaryOp):
        val = fold_divmod(val)
    return fold_mul_chains(val)


def add_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    """`add`. A register += immediate follows the same path as MIPS
    `addu reg, reg, imm` (handle_add): on pointers, handle_addi_real's
    field-path resolution recovers `&s->field` / array-element addresses
    instead of leaving raw arithmetic."""
    src = srcs[0]
    if isinstance(a.raw_arg(0), Register) and isinstance(src, Literal):
        dst = a.reg_ref(0)
        return fold_mul_chains(handle_addi_real(dst, dst, lhs, src, a))
    return handle_add_real(lhs, src, a)


def carry_in(a: InstrArgs) -> Expression:
    """The x86 carry flag as a 0/1 integer expression (for adc/sbb)."""
    return condition_from_expr(a.regs[Register("c")])


def adc_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    return handle_add_real(handle_add_real(lhs, srcs[0], a), carry_in(a), a)


def sbb_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    if a.raw_arg(0) == a.raw_arg(1):
        # sbb r, r: idiom for materializing the carry (borrow) flag as
        # 0 / -1 without branching. After `cmp a, b` the carry is the unsigned
        # borrow (a < b), so this is -(a < b); `neg`/`inc` of it (below)
        # recover the plain (in)equality.
        return UnaryOp("-", carry_in(a), type=Type.intish())
    return BinaryOp.intptr(handle_sub(lhs, srcs[0]), "-", carry_in(a))


def neg_expr(v: Expression) -> Expression:
    """`neg`. Collapses -(-x) (always valid in two's complement), so
    `sbb r,r; neg r` reads as the plain comparison."""
    uw = early_unwrap(v)
    if isinstance(uw, UnaryOp) and uw.op == "-" and not uw.expr.type.is_float():
        return uw.expr
    return UnaryOp.sint("-", v)


def inc_expr(a: InstrArgs, v: Expression) -> Expression:
    """`inc`. 1 + -(cond) = !cond, so `sbb r,r; inc r` reads as the negated
    comparison; the fold only fires on a negated comparison."""
    uw = early_unwrap(v)
    if isinstance(uw, UnaryOp) and uw.op == "-":
        inner = early_unwrap(uw.expr)
        if isinstance(inner, BinaryOp) and inner.is_comparison():
            return inner.negated()
    return handle_add_real(v, Literal(1), a)


def shrd_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    src, amt = srcs
    if isinstance(amt, Literal) and 0 < amt.value < 32:
        return BinaryOp.int(
            BinaryOp.uint(lhs, ">>", amt),
            "|",
            fold_mul_chains(BinaryOp.int(src, "<<", Literal(32 - amt.value))),
        )
    return fn_op("M2C_SHRD", [lhs, src, amt], Type.u32())


def shld_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    src, amt = srcs
    if isinstance(amt, Literal) and 0 < amt.value < 32:
        return BinaryOp.int(
            fold_mul_chains(BinaryOp.int(lhs, "<<", amt)),
            "|",
            BinaryOp.uint(src, ">>", Literal(32 - amt.value)),
        )
    return fn_op("M2C_SHLD", [lhs, src, amt], Type.u32())


# --- MSVC CRT 64-bit integer helpers ---------------------------------------
#
# MSVC6 lowers `__int64`/`unsigned __int64` arithmetic and shifts into calls to
# private-convention CRT routines that do NOT follow cdecl. Two distinct ABIs
# occur (both observed in MSVC6 /O2 output):
#
#  - multiply/divide/remainder (__allmul/__alldiv/__aulldiv/__allrem/
#    __aullrem): the two 64-bit operands are passed on the stack as four
#    dwords (a_lo, a_hi, b_lo, b_hi -- i.e. two ordinary cdecl 8-byte slots, at
#    [esp+4] and [esp+0xc]), and the callee pops all 16 argument bytes itself
#    (stdcall-like). The 64-bit result comes back in edx:eax (edx = high).
#
#  - shifts (__allshl/__allshr/__aullshr): the 64-bit value is passed in
#    edx:eax (edx = high, eax = low) and the shift count in cl/ecx; there are
#    NO stack arguments and nothing to clean up. The result is in edx:eax.
#
# Each maps directly onto a real 64-bit IR op (`*`, `/`, `%`, `<<`, `>>`), with
# the two edx:eax pairs reconstructed by glue_int64 and the result modeled the
# same way a 64-bit-returning callee is (eax = whole value, edx = SECOND_REG).
# (`__ftol`, the double->long cast helper, is handled separately via the x87
# call-delta mechanism; the combined div+rem __alldvrm/__aulldvrm helpers are
# not emitted by MSVC6 in practice and are left to degrade to a plain call.)
@dataclass(frozen=True)
class MathHelper:
    kind: str  # "muldiv" (two stack operands) or "shift" (edx:eax)
    op: str  # the C operator to emit
    signed: bool  # operand/result signedness
    cleanup: int  # stack bytes the callee pops on return


X86_MATH_HELPERS: Dict[str, MathHelper] = {
    "__allmul": MathHelper("muldiv", "*", True, 16),
    "__alldiv": MathHelper("muldiv", "/", True, 16),
    "__aulldiv": MathHelper("muldiv", "/", False, 16),
    "__allrem": MathHelper("muldiv", "%", True, 16),
    "__aullrem": MathHelper("muldiv", "%", False, 16),
    "__allshl": MathHelper("shift", "<<", True, 0),
    "__allshr": MathHelper("shift", ">>", True, 0),
    "__aullshr": MathHelper("shift", ">>", False, 0),
}


def _second_reg_source(expr: Expression) -> Optional[Expression]:
    """If `expr` is a `SECOND_REG(x)` marker -- m2c's placeholder for the high
    half of a 64-bit value whose whole value already lives in the paired low
    register (see function_return) -- return x, else None."""
    uw = early_unwrap(expr)
    if (
        isinstance(uw, FuncCall)
        and isinstance(uw.function, GlobalSymbol)
        and uw.function.symbol_name == "SECOND_REG"
        and len(uw.args) == 1
    ):
        return uw.args[0]
    return None


def glue_int64(lo: Expression, hi: Expression, *, signed: bool) -> Expression:
    """Reconstruct the 64-bit value held in a (low, high) 32-bit register/slot
    pair as a single s64/u64 expression. When the high half is `SECOND_REG(x)`
    -- edx:eax already carrying a whole 64-bit x, m2c's register-pair
    convention for a value produced by a prior 64-bit op -- recover x directly;
    otherwise splice the halves as `((u64)hi << 32) | (u32)lo`."""
    tp = Type.s64() if signed else Type.u64()
    src = _second_reg_source(hi)
    if src is not None:
        return as_type(src, tp, silent=True)
    lo_u = as_type(lo, Type.u32(), silent=True)
    hi_u = as_type(as_type(hi, Type.u32(), silent=True), Type.u64(), silent=True)
    shifted = BinaryOp(left=hi_u, op="<<", right=Literal(32), type=Type.u64())
    glued = BinaryOp(left=shifted, op="|", right=lo_u, type=Type.u64())
    return as_type(glued, tp, silent=True)


def eval_math_helper(
    spec: MathHelper,
    s: NodeState,
    a: InstrArgs,
    arg_base: Optional[int],
) -> bool:
    """Model a call to a 64-bit CRT math helper as its real IR op, placing the
    64-bit result in edx:eax (eax = whole value, edx = SECOND_REG, mirroring a
    64-bit-returning callee). Returns False -- so the caller falls back to a
    plain call -- when the operands cannot be recovered (a `muldiv` helper
    whose stack argument slots are not all present)."""
    tp = Type.s64() if spec.signed else Type.u64()
    if spec.kind == "shift":
        # Value in edx:eax, count in ecx.
        value = glue_int64(a.regs[EAX], a.regs[EDX], signed=spec.signed)
        count = as_intish(a.regs[ECX])
        result: Expression = BinaryOp(left=value, op=spec.op, right=count, type=tp)
    else:
        # Two 64-bit operands, each two dwords, on the stack: a at arg_base+0/+4,
        # b at arg_base+8/+12 (see the ABI note on X86_MATH_HELPERS).
        if arg_base is None:
            return False
        slots = [s.subroutine_args.get(arg_base + off) for off in (0, 4, 8, 12)]
        if any(slot is None for slot in slots):
            return False
        a_lo, a_hi, b_lo, b_hi = slots
        assert a_lo is not None and a_hi is not None
        assert b_lo is not None and b_hi is not None
        lhs = glue_int64(a_lo, a_hi, signed=spec.signed)
        rhs = glue_int64(b_lo, b_hi, signed=spec.signed)
        result = BinaryOp(left=lhs, op=spec.op, right=rhs, type=tp)
        for off in (0, 4, 8, 12):
            s.subroutine_args.pop(arg_base + off, None)
    s.clear_caller_save_regs()
    s.set_reg(EAX, result)
    s.set_reg(EDX, fn_op("SECOND_REG", [result], Type.reg32(likely_float=False)))
    return True


# Disassembler exports encode stdcall decoration in symbol names: `_name@8` becomes
# `__imp__name_8` for import thunks. The trailing number is the number of
# argument bytes the callee pops on return.
RE_STDCALL_IMPORT = re.compile(r"^__imp__.*_(\d+)$")


def is_register_indirect_call(target: Argument) -> bool:
    """Whether a call target is an indirect call through a register (a COM /
    virtual method call, e.g. `call eax` or `call [ecx + 0x7c]`), as opposed
    to a direct call or a call through an absolute import slot."""
    if isinstance(target, Register):
        return True
    if isinstance(target, AsmAddressMode) and target.base != ZERO:
        return True
    return False


def callee_cleanup_bytes(
    target: Argument,
    context_arg_bytes: Optional[Dict[str, int]] = None,
    file_arg_bytes: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    """Number of stack bytes a call target is known to pop itself: 0 for a
    known-cdecl callee, None when the convention cannot be determined from the
    name. The sources, in strict precedence order, are marked 0-2 below;
    structural inference (compute_call_cleanup) runs only when this returns
    None, validated by the esp-balance check at return."""
    context_arg_bytes = context_arg_bytes or {}
    file_arg_bytes = file_arg_bytes or {}
    sym = call_target_symbol(target)
    if sym is None:
        return None
    # 0. MSVC CRT 64-bit math helpers have a fixed private ABI (the mul/div/rem
    # helpers pop their 16 argument bytes; the register-argument shift helpers
    # pop nothing). This is authoritative and independent of any name decoration.
    spec = X86_MATH_HELPERS.get(sym)
    if spec is not None:
        return spec.cleanup
    # 1. Explicit context/prototype (overrides everything below).
    if sym in context_arg_bytes:
        return context_arg_bytes[sym]
    # 2. Decorated stdcall suffix: inline `__imp__X_N`, then file `.set name@N`.
    m = RE_STDCALL_IMPORT.match(sym)
    if m:
        return int(m.group(1))
    if sym in file_arg_bytes:
        return file_arg_bytes[sym]
    return None


# The ESP-delta dataflow state at one program point: the (nonpositive) offset
# of esp relative to its value at function entry, plus the offset ebp holds
# when it is a copy of esp (`push ebp; mov ebp, esp` frames), or None if ebp
# holds an unrelated value.
EspState = Tuple[int, Optional[int]]


# Callee-saved registers, whose push/pop pairs are register saves (never call
# arguments). The cdecl scratch registers eax/ecx/edx are deliberately excluded
# (an `add esp`/`pop ecx` cleanup after a call restores them, and they are the
# regs actually pushed as arguments), so they never spoof a save/restore pair.
SAVED_PUSH_REGS = {EBX, ESI, EDI, EBP}


def compute_save_pushes(body: List[BodyPart]) -> Set[int]:
    """Indices of prologue `push` instructions that save a register or the
    frame pointer, rather than pushing an outgoing call argument. Computed
    purely structurally (before the ESP dataflow runs), so the stdcall/indirect
    cleanup inference does not miscount prologue and callee-save pushes as call
    arguments.

    Register saves live in the prologue: the initial run of frame setup
    (`sub esp`, `push ebp; mov ebp, esp`, callee-saved pushes) before the first
    call/branch/return or non-save push. Only callee-saved registers that are
    restored later, and each at most once, count -- so that a callee-saved
    register pushed again as a call argument (a real pattern, e.g. `push edi`
    to pass a pointer while edi is also a saved register) is left as an
    argument, and an entry `push esi` that is passed as an argument to the very
    first call (never restored) is not mistaken for a save."""
    restored: Set[Register] = set()
    for part in body:
        if isinstance(part, Instruction):
            base, _ = split_width_suffix(part.mnemonic)
            if base == "pop" and part.args and part.args[0] in SAVED_PUSH_REGS:
                assert isinstance(part.args[0], Register)
                restored.add(part.args[0])

    def is_frame_ptr_save(index: int) -> bool:
        nxt = next((p for p in body[index + 1 :] if isinstance(p, Instruction)), None)
        return (
            nxt is not None
            and split_width_suffix(nxt.mnemonic)[0] == "mov"
            and len(nxt.args) == 2
            and nxt.args[0] == EBP
            and nxt.args[1] == ESP
        )

    saves: Set[int] = set()
    saved_regs: Set[Register] = set()
    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            continue  # labels are transparent within the prologue
        base, _ = split_width_suffix(part.mnemonic)
        arg0 = part.args[0] if part.args else None
        if part.mnemonic == "push":
            if arg0 == EBP and is_frame_ptr_save(i):
                # The frame-pointer save; its restore is `leave`'s implicit pop
                # (no explicit `pop ebp`), so it is not in `restored`.
                saves.add(i)
                continue
            if (
                isinstance(arg0, Register)
                and arg0 in SAVED_PUSH_REGS
                and arg0 in restored
                and arg0 not in saved_regs
            ):
                saves.add(i)
                saved_regs.add(arg0)
                continue
            break  # a non-save push: the prologue's argument region begins.
        if (
            part.function_target is not None
            or part.jump_target is not None
            or part.is_return
            or part.is_conditional
        ):
            break  # the first call/branch/return ends the prologue.
    return saves


CHKSTK_NAMES = {"__chkstk", "_chkstk", "__alloca_probe", "_alloca_probe"}


class X86ChkstkPattern(AsmPattern):
    """MSVC emits `mov eax, N; call __chkstk` instead of `sub esp, N` for
    frames larger than a page (__chkstk touches each new stack page in order
    to trigger guard-page growth). Semantically the pair is exactly a frame
    allocation; rewrite it into the `sub` so the ESP-delta pass sees it."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        parts = matcher.input[matcher.index : matcher.index + 2]
        if len(parts) < 2:
            return None
        mov, call = parts
        if (
            isinstance(mov, Instruction)
            and isinstance(call, Instruction)
            and mov.mnemonic == "mov"
            and mov.args[0] == EAX
            and isinstance(mov.args[1], AsmLiteral)
            and call.mnemonic == "call"
            and isinstance(call.args[0], AsmGlobalSymbol)
            and call.args[0].symbol_name in CHKSTK_NAMES
        ):
            sub = AsmInstruction("sub", [ESP, AsmLiteral(mov.args[1].value)])
            return Replacement([sub], 2, clobbers=[EAX])
        return None


class X86PushAllocPattern(AsmPattern):
    """MSVC allocates a single 4-byte local with `push <scratch>` (one byte)
    rather than `sub esp, 4` (three bytes), releasing it before `ret`. At a
    __cdecl/__stdcall entry the pushed scratch register (eax/ecx/edx) holds no
    live value, so the bracket is exactly a frame allocate/deallocate; rewrite
    it to `sub esp, 4` / `add esp, 4` so no register value is stored.

    The dealloc before each `ret` is either the matching `pop <same reg>`
    (rewritten to `add esp, 4`), or -- when MSVC batches the local's release
    into the caller-side cleanup of deferred stack arguments -- a plain
    `add esp, N` (N >= 4) that already includes the local's 4 bytes and is
    left untouched. Only these two forms count: a real value push (a
    __fastcall/__thiscall argument spill) is paired with neither."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        if matcher.index != 0:
            return None
        body = matcher.input
        push_idx = next(
            (i for i, p in enumerate(body) if isinstance(p, Instruction)), None
        )
        if push_idx is None:
            return None
        push = body[push_idx]
        assert isinstance(push, Instruction)
        if not (
            push.mnemonic == "push"
            and len(push.args) == 1
            and push.args[0] in (EAX, ECX, EDX)
        ):
            return None
        reg = push.args[0]

        # Every `ret` must be immediately preceded (skipping labels) by a
        # deallocation of the local: `pop <reg>` (rewritten to `add esp, 4`) or
        # `add esp, N` with N >= 4 (left as-is). There must be at least one.
        dealloc_pops: Set[int] = set()
        found_dealloc = False
        prev_idx: Optional[int] = None
        for i, part in enumerate(body):
            if isinstance(part, Label):
                continue
            assert isinstance(part, Instruction)
            if part.is_return:
                if prev_idx is None:
                    return None
                prev = body[prev_idx]
                if not isinstance(prev, Instruction):
                    return None
                is_pop = (
                    prev.mnemonic == "pop"
                    and len(prev.args) == 1
                    and prev.args[0] == reg
                )
                is_add = (
                    prev.mnemonic == "add"
                    and len(prev.args) == 2
                    and prev.args[0] == ESP
                    and isinstance(prev.args[1], AsmLiteral)
                    and prev.args[1].value >= 4
                )
                if not (is_pop or is_add):
                    return None
                if is_pop:
                    dealloc_pops.add(prev_idx)
                found_dealloc = True
            prev_idx = i
        if not found_dealloc:
            return None

        new_body: List[ReplacementPart] = []
        for i, part in enumerate(body):
            if i == push_idx:
                new_body.append(AsmInstruction("sub", [ESP, AsmLiteral(4)]))
            elif i in dealloc_pops:
                new_body.append(AsmInstruction("add", [ESP, AsmLiteral(4)]))
            else:
                new_body.append(part)
        return Replacement(new_body, len(body), clobbers=[reg])


class X86SehPattern(SimpleAsmPattern):
    """MSVC's structured exception handling bookkeeping. After the
    `push ebp; mov ebp, esp` frame setup, an SEH-using function pushes a
    16-byte exception registration record and installs it at fs:[0]:

        push -0x1                    # trylevel
        push <scopetable>
        push <handler>               # e.g. __except_handler3
        mov eax, fs:[0]              # old head of the handler chain
        push eax
        mov fs:[0], esp              # install the record

    and the epilogue stores the saved head back to fs:[0]. None of this is
    visible in the function's C-level semantics (__except blocks are entered
    only through the SEH dispatcher), so the prologue is replaced by a bare
    16-byte frame allocation and the epilogue store is dropped. Nonstandard
    fs: accesses are left alone and fail translation with a clear error."""

    pattern = make_pattern(
        "push N",
        "push _",
        "push _",
        "mov.fs $x, 0($zero)",
        "push $x",
        "mov.fs 0($zero), $esp",
    )

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        handler = m.body[2]
        assert isinstance(handler, Instruction)
        if not (
            m.literals["N"] in (-1, 0xFFFFFFFF)
            and isinstance(handler.args[0], AsmGlobalSymbol)
            and "except_handler" in handler.args[0].symbol_name
        ):
            return None
        sub = AsmInstruction("sub", [ESP, AsmLiteral(16)])
        return Replacement([sub], len(m.body))


class X86SehEpiloguePattern(SimpleAsmPattern):
    """Remove stores that restore or update the canonical fs:[0] SEH head."""

    pattern = make_pattern("mov.fs 0($zero), $x")

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        if m.regs["x"] == ESP:
            return None
        return Replacement([], len(m.body), clobbers=[])


class X86RawJumpTablePattern(AsmPattern):
    """Reject raw-address jump tables, independently of label spelling."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        part = matcher.input[matcher.index]
        if (
            not isinstance(part, Instruction)
            or part.mnemonic != "jmp"
            or not isinstance(part.args[0], AsmAddressMode)
        ):
            return None
        addr = part.args[0]
        # A jump table has an indexed address with pointer-sized (4-byte)
        # entries. Requiring this scaled-index term avoids mistaking
        # `jmp [absolute_address]` tail calls for switches.
        has_scaled_index = any(
            isinstance(sub, BinOp)
            and sub.op == "*"
            and isinstance(sub.lhs, Register)
            and isinstance(sub.rhs, AsmLiteral)
            and sub.rhs.value == 4
            for sub in traverse_arg(addr.addend)
        )
        if not has_scaled_index:
            return None
        # Symbolic tables are handled by the normal switch machinery. A raw
        # address cannot be associated with input data without address/section
        # metadata, so reject it explicitly instead of guessing from labels.
        literals = [
            sub for sub in traverse_arg(addr.addend) if isinstance(sub, AsmLiteral)
        ]
        table_addrs = [lit for lit in literals if lit.value > 0xFFFF]
        if len(table_addrs) != 1:
            return None
        raise DecompFailure(
            "raw-address jump table is unsupported; give the table a symbol "
            "and use that symbol in the jump operand"
        )


class X86RewritePattern(AsmPattern):
    """Whole-body rewrite eliminating x86's moving integer and x87 stacks.

    m2c's stack machinery assumes a stack pointer that is fixed after the
    prologue, which x86 code violates constantly (push/pop, caller-side
    argument pushes, `add esp, N` cleanup). This pass runs a linear dataflow
    analysis over the body computing the ESP delta from function entry at
    every instruction, then rewrites all stack operations into accesses at
    fixed offsets from a virtual frame base (esp at its deepest point):

    - the frame size is the maximum stack depth; a synthetic `sub esp, N` is
      inserted at the top so that get_stack_info sees the full frame;
    - `[esp + k]` at delta d becomes `[esp + k + frame_size + d]`, and
      `[ebp + k]` similarly resolves through the tracked ebp copy;
    - `push reg` with a matching `pop reg` of the same slot (callee-saved
      register saves, mid-function spills) becomes a plain store/load pair;
    - other `push`es become `storearg.fictive` argument stores that feed the
      next `call`'s argument list;
    - prologue `push ecx`-style frame allocation and all `sub/add esp, N`
      become pure delta bookkeeping (no emitted instruction);
    - each `call` gets its current argument-region base (and its argument
      byte count, when a cleanup `add esp, N`/`pop ecx` follows or the callee
      is a known-stdcall import) appended as literal args;
    - `jmp` to a label outside the function becomes `call; ret`.
    """

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        if matcher.index != 0:
            return None
        context_facts = (
            compute_x86_context_facts(matcher.typemap)
            if matcher.typemap is not None
            else EMPTY_X86_CONTEXT_FACTS
        )
        try:
            new_body = rewrite_stack_ops(
                matcher.input,
                matcher.arch,
                matcher.asm_data,
                matcher.labels,
                context_facts,
            )
        except DecompFailure as e:
            # An inconsistent dataflow usually means an undecorated stdcall
            # callee (which pops its own arguments) was taken for cdecl.
            # Retry with structural stdcall inference enabled for direct
            # calls, and keep the result only if the dataflow then becomes
            # consistent; otherwise re-raise the original error. Functions
            # that analyze fine without inference are never affected.
            try:
                new_body = rewrite_stack_ops(
                    matcher.input,
                    matcher.arch,
                    matcher.asm_data,
                    matcher.labels,
                    context_facts,
                    infer_direct_stdcall=True,
                )
            except DecompFailure:
                raise e from None
        new_body = rewrite_fpu_stack(
            new_body,
            matcher.arch,
            matcher.asm_data,
            context_facts.fpu_call_deltas,
        )
        return Replacement(new_body, len(matcher.input), clobbers=[])


def rewrite_stack_ops(
    body: List[BodyPart],
    arch: ArchAsm,
    asm_data: AsmData,
    labels: Set[str],
    context_facts: Optional[X86ContextFacts] = None,
    *,
    infer_direct_stdcall: bool = False,
) -> List[BodyPart]:
    label_pos: Dict[str, int] = {}
    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            for name in part.names:
                label_pos[name] = i

    def instr_str(item: Instruction) -> str:
        return f"`{item}` {item.meta.loc_str()}"

    # How many argument bytes each call's callee pops from the stack. cdecl
    # callees pop nothing (the caller emits an `add esp, N`/`pop` cleanup);
    # stdcall callees pop their arguments. MSVC's stdcall imports are not
    # always name-decorated (`__imp__X_8`), so this is also inferred
    # structurally: a call whose pushed arguments are not cleaned up by the
    # caller must be to a stdcall callee that popped them itself.
    ESP_NEUTRAL_STOP = {
        "call",
        "push",
        "pop",
        "pushad",
        "popad",
        "jmp",
        "ret",
        "leave",
        "storearg.fictive",
    }

    # Identified structurally, before the ESP dataflow exists; see
    # compute_save_pushes for why the argument scan must not see these.
    save_pushes = compute_save_pushes(body)

    def call_arg_bytes(call_index: int) -> int:
        """Bytes pushed as outgoing arguments for the call at `call_index`,
        found by scanning back over the contiguous run of arg-building
        instructions (pushes interleaved with esp-neutral setup). Labels are
        transparent: an argument may be pushed just before a merge label whose
        incoming paths each push it (e.g. `push x; jmp L` / `push y; L: call`).
        Register-save / frame pushes are not arguments and stop the scan."""
        total = 0
        j = call_index - 1
        while j >= 0:
            part = body[j]
            if not isinstance(part, Instruction):
                j -= 1
                continue
            base, _ = split_width_suffix(part.mnemonic)
            if part.mnemonic == "push":
                if j in save_pushes:
                    break  # a callee-save / frame-pointer save, not an argument.
                total += 4
            elif base == "push":
                break  # sub-word push; give up.
            elif (
                base == "mov"
                and len(part.args) == 2
                and part.args[0] == EBP
                and part.args[1] == ESP
            ):
                break  # `mov ebp, esp` frame setup: nothing before it is an arg.
            elif (
                base in ESP_NEUTRAL_STOP
                or part.function_target is not None
                or part.jump_target is not None
                or part.is_conditional
                or part.is_return
            ):
                break  # a call/branch/return barrier bounds the arg region.
            elif ESP in part.outputs or ESP in part.clobbers:
                break
            j -= 1
        return total

    # Callee cleanup information beyond inline name decoration, kept in
    # precedence tiers (see callee_cleanup_bytes): user-context stdcall
    # prototypes (highest), then file-level `.set sym, "name@N"` metadata.
    context_arg_bytes = (
        dict(context_facts.stdcall_arg_bytes) if context_facts is not None else {}
    )
    file_arg_bytes: Dict[str, int] = dict(asm_data.stdcall_arg_bytes)

    call_cleanup: Dict[int, int] = {}

    def caller_cleans_up(call_index: int) -> bool:
        """Whether the caller restores esp for this call's arguments: either
        a cleanup (`add esp, N` / `pop ecx`) directly after the call, or an
        MSVC batched cleanup further down the same straight-line region -- a
        single `add esp, N` covering several calls' arguments at once, which
        must include this call's if it pops more than the argument bytes
        pushed after this call returned."""
        extra = 0  # argument bytes pushed since the call returned
        j = call_index + 1
        while j < len(body):
            part = body[j]
            if not isinstance(part, Instruction):
                return False  # a label: control flow merge; give up.
            base, _ = split_width_suffix(part.mnemonic)
            if (
                base == "add"
                and part.args[0] == ESP
                and isinstance(part.args[1], AsmLiteral)
            ):
                if part.args[1].value > extra:
                    return True
                extra -= part.args[1].value
            elif base == "pop" and part.args[0] in (ECX, EDX):
                if extra == 0:
                    return True
                extra -= 4
            elif part.mnemonic == "push":
                extra += 4
            elif base == "call" and part.function_target is not None:
                # Nested call; account for what its callee pops (computed
                # already: calls are classified back to front).
                extra -= call_cleanup.get(j, 0)
                if extra < 0:
                    # A later stdcall callee popped more than was pushed
                    # after this call, so the pushes taken for this call's
                    # arguments belonged to that callee instead.
                    return True
            elif (
                base in ESP_NEUTRAL_STOP
                or part.function_target is not None
                or part.is_return
                or part.jump_target is not None
                or part.is_conditional
                or ESP in part.outputs
                or ESP in part.clobbers
            ):
                return False
            j += 1
        return False

    def compute_call_cleanup(call_index: int) -> int:
        call = body[call_index]
        assert isinstance(call, Instruction)
        target = call.args[0]
        known = callee_cleanup_bytes(target, context_arg_bytes, file_arg_bytes)
        if known is not None:
            return known
        # The callee's convention is not known from its name. stdcall callees
        # pop their own arguments; cdecl callees (including MSVC's
        # batched-cleanup pattern, where one `add esp, N` restores several
        # calls' arguments at once) always have the caller restore esp. We
        # treat a call as self-cleaning when no caller cleanup follows and
        # either:
        #  - it is an indirect call through a register (`call [ecx+0x7c]`,
        #    `call eax`), i.e. a COM/virtual method, or
        #  - `infer_direct_stdcall` is set: the retry pass extends the same
        #    inference to direct calls (undecorated stdcall callees), relying
        #    on the dataflow consistency check to validate the result.
        if is_register_indirect_call(target) and not caller_cleans_up(call_index):
            return call_arg_bytes(call_index)
        if (
            infer_direct_stdcall
            and call_target_symbol(target) is not None
            and not caller_cleans_up(call_index)
        ):
            return call_arg_bytes(call_index)
        return 0

    # Classify calls back to front so that the forward scan in
    # caller_cleans_up can account for later callees' stack pops.
    for i in reversed(range(len(body))):
        part = body[i]
        if isinstance(part, Instruction) and part.function_target is not None:
            base, _ = split_width_suffix(part.mnemonic)
            if base == "call":
                call_cleanup[i] = compute_call_cleanup(i)

    # The esp delta ebp holds after the function's `mov ebp, esp` frame setup.
    # `pushad`/`popad` and mid-function scratch use of ebp can lose the tracked
    # ebp value on a path; recovering it from this frame constant lets the
    # epilogue's `mov esp, ebp`/`leave` still resolve. Ambiguous (conflicting)
    # setups leave it None, disabling the fallback.
    frame_ebp: Optional[int] = None
    frame_ebp_ambiguous = False

    # Pass 1: dataflow analysis computing the ESP delta at entry to every
    # reachable instruction.
    states: List[Optional[EspState]] = [None] * len(body)

    def merge(a: EspState, b: EspState, item: Instruction) -> EspState:
        if a[0] != b[0]:
            raise DecompFailure(
                f"x86 stack analysis failed: conflicting stack depths "
                f"({-a[0]:#x} vs {-b[0]:#x}) at {instr_str(item)}"
            )
        return (a[0], a[1] if a[1] == b[1] else None)

    def step(item: Instruction, st: EspState, index: int) -> None:
        """Push the out-state(s) of `item` onto the worklist."""
        nonlocal frame_ebp, frame_ebp_ambiguous
        esp, ebp = st
        base, _ = split_width_suffix(item.mnemonic)
        args = item.args
        fallthrough = True

        def jump_to(name: str, target_st: EspState) -> None:
            pos = label_pos.get(name)
            if pos is not None:
                worklist.append((pos, target_st))

        if item.is_return:
            # esp must be back at its entry value before a return (a stdcall
            # `ret N` pops caller-frame bytes above it). A nonzero delta means
            # some callee's cleanup was misclassified (stdcall vs cdecl); fail
            # loud so the infer_direct_stdcall retry can correct it.
            if esp != 0:
                raise DecompFailure(
                    f"x86 stack analysis failed: esp is {-esp:#x} bytes from its "
                    f"entry value at {instr_str(item)}, not balanced for return. "
                    f"This usually means a called function's argument cleanup was "
                    f"misclassified (stdcall vs cdecl)."
                )
            return
        if base == "push":
            if item.mnemonic != "push":
                raise DecompFailure(f"unsupported sub-word x86 push: {instr_str(item)}")
            st = (esp - 4, ebp)
        elif base == "pop":
            st = (esp + 4, None if args[0] == EBP else ebp)
        elif base == "pushad":
            st = (esp - 32, ebp)
        elif base == "popad":
            # popad restores ebp to its value at the matching pushad, which
            # for a frame pointer is the frame value.
            st = (esp + 32, frame_ebp if not frame_ebp_ambiguous else None)
        elif base in ("sub", "add") and args[0] == ESP:
            if not isinstance(args[1], AsmLiteral):
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
            st = (esp - args[1].value if base == "sub" else esp + args[1].value, ebp)
        elif base == "mov" and args[0] == EBP and args[1] == ESP:
            if frame_ebp is not None and frame_ebp != esp:
                frame_ebp_ambiguous = True
            frame_ebp = esp
            st = (esp, esp)
        elif base == "mov" and args[0] == ESP:
            src_ebp = (
                ebp
                if ebp is not None
                else (frame_ebp if not frame_ebp_ambiguous else None)
            )
            if args[1] != EBP or src_ebp is None:
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
            st = (src_ebp, src_ebp)
        elif base == "lea" and args[0] == ESP:
            src = args[1]
            if (
                isinstance(src, AsmAddressMode)
                and isinstance(src.addend, AsmLiteral)
                and (src.base == ESP or (src.base == EBP and ebp is not None))
            ):
                src_esp = esp if src.base == ESP else ebp
                assert src_esp is not None
                st = (src_esp + src.addend.value, ebp)
            else:
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
        elif base == "leave":
            src_ebp = (
                ebp
                if ebp is not None
                else (frame_ebp if not frame_ebp_ambiguous else None)
            )
            if src_ebp is None:
                raise DecompFailure(
                    f"`leave` without a tracked ebp frame: {instr_str(item)}"
                )
            st = (src_ebp + 4, None)
        elif base == "and" and args[0] == ESP:
            # Stack alignment (`and esp, -8`). The rewritten frame is an
            # abstraction anyway, so treat alignment as a no-op; other masks
            # are rejected below.
            if not (
                isinstance(args[1], AsmLiteral)
                and args[1].value & 0xFFFFFFFF in (0xFFFFFFF0, 0xFFFFFFF8, 0xFFFFFFFC)
            ):
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
        elif item.function_target is not None:
            st = (esp + call_cleanup.get(index, 0), ebp)
        elif base == "jmp":
            if isinstance(item.jump_target, JumpTarget):
                if item.jump_target.target in label_pos:
                    jump_to(item.jump_target.target, st)
                # else: a tail call; terminal.
                return
            # Indirect jump: a jump table.
            targets = switch_jump_table_labels(item, asm_data)
            if targets is None:
                raise DecompFailure(
                    f"Unable to determine jump table for {instr_str(item)}"
                )
            for target in targets:
                jump_to(target, st)
            return
        else:
            if ESP in item.outputs:
                raise DecompFailure(
                    f"cannot statically analyze stack adjustment {instr_str(item)}"
                )
            if EBP in item.outputs or EBP in item.clobbers:
                st = (esp, None)
        if isinstance(item.jump_target, JumpTarget):
            jump_to(item.jump_target.target, st)
            if not item.is_conditional:
                fallthrough = False
        if fallthrough:
            worklist.append((index + 1, st))

    worklist: List[Tuple[int, EspState]] = [(0, (0, None))]
    while worklist:
        index, st = worklist.pop()
        if index >= len(body):
            continue
        part = body[index]
        if not isinstance(part, Instruction):
            # Labels pass the state through.
            prev = states[index]
            if prev is not None:
                st = (st[0], st[1] if st[1] == prev[1] else None)
                if st[0] != prev[0]:
                    raise DecompFailure(
                        f"x86 stack analysis failed: conflicting stack depths "
                        f"({-st[0]:#x} vs {-prev[0]:#x}) at label {part}"
                    )
                if st == prev:
                    continue
            states[index] = st
            worklist.append((index + 1, st))
            continue
        prev = states[index]
        if prev is not None:
            st = merge(prev, st, part)
            if st == prev:
                continue
        states[index] = st
        step(part, st, index)

    # The frame is a fixed-size region covering the deepest stack extent.
    frame_size = 0
    for i, part in enumerate(body):
        st = states[i]
        if st is None or not isinstance(part, Instruction):
            continue
        depth = -st[0]
        base, _ = split_width_suffix(part.mnemonic)
        if base == "push":
            depth += 4
        elif base == "pushad":
            depth += 32
        elif base == "sub" and part.args[0] == ESP:
            assert isinstance(part.args[1], AsmLiteral)
            depth += part.args[1].value
        frame_size = max(frame_size, depth)

    # Pass 2: classify pops (cleanup pops directly after calls discard
    # argument slots) and pushes (saves/spills with a matching pop, prologue
    # frame allocations, and call argument stores).
    cleanup_pops: Set[int] = set()
    prev_instr: Optional[Instruction] = None
    prev_index = -1
    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            prev_instr = None
            continue
        base, _ = split_width_suffix(part.mnemonic)
        if (
            base == "pop"
            and part.args[0] in (ECX, EDX)
            and prev_instr is not None
            and (prev_instr.function_target is not None or prev_index in cleanup_pops)
        ):
            cleanup_pops.add(i)
        prev_instr = part
        prev_index = i

    def cdecl_arg_bytes(call_index: int) -> int:
        """For a cdecl call: the number of argument bytes attributable to
        this call based on the caller-side cleanup that follows, or -1 when
        no cleanup is found (translation then takes all pending argument
        stores). Handles immediate cleanup (`add esp, N`, cleanup pops),
        MSVC batched cleanup (one add covering several calls' arguments),
        and argument regions consumed by a later stdcall callee (in which
        case the pushes preceding this call belonged to that callee)."""
        consume = -1
        extra = 0  # argument bytes pushed since this call returned
        j = call_index + 1
        while j < len(body):
            part = body[j]
            if not isinstance(part, Instruction):
                break
            base, _ = split_width_suffix(part.mnemonic)
            if (
                base == "add"
                and part.args[0] == ESP
                and isinstance(part.args[1], AsmLiteral)
            ):
                n = part.args[1].value
                if n > extra:
                    return (0 if consume < 0 else consume) + n - extra
                extra -= n
            elif j in cleanup_pops:
                if extra == 0:
                    consume = (0 if consume < 0 else consume) + 4
                else:
                    extra -= 4
            elif part.mnemonic == "push":
                extra += 4
            elif base == "call" and part.function_target is not None:
                extra -= call_cleanup.get(j, 0)
                if extra < 0:
                    # A later stdcall callee popped more than was pushed
                    # after this call: the remaining pushes were its
                    # arguments, not this call's.
                    return 0 if consume < 0 else consume
            elif (
                base in ESP_NEUTRAL_STOP
                or part.function_target is not None
                or part.is_return
                or part.jump_target is not None
                or part.is_conditional
                or ESP in part.outputs
                or ESP in part.clobbers
            ):
                break
            j += 1
        return consume

    # Locations (offsets from the virtual frame base) that non-cleanup pops
    # read back into the same register, plus `leave`'s implicit pop of ebp.
    pop_locs: Set[Tuple[int, Register]] = set()
    for i, part in enumerate(body):
        st = states[i]
        if not isinstance(part, Instruction) or st is None or i in cleanup_pops:
            continue
        base, _ = split_width_suffix(part.mnemonic)
        if base == "pop" and isinstance(part.args[0], Register):
            pop_locs.add((frame_size + st[0], part.args[0]))
        elif base == "leave" and st[1] is not None:
            pop_locs.add((frame_size + st[1], EBP))

    def is_save_push(index: int) -> bool:
        part = body[index]
        st = states[index]
        assert isinstance(part, Instruction) and st is not None
        arg = part.args[0]
        return isinstance(arg, Register) and (frame_size + st[0] - 4, arg) in pop_locs

    # The prologue: the initial run of frame-establishing instructions.
    # Register pushes within it that are never popped back allocate frame
    # space (MSVC's `push ecx` idiom); their stored value is meaningless.
    alloc_pushes: Set[int] = set()
    last_frame_op = -1
    for i, part in enumerate(body):
        if not isinstance(part, Instruction) or states[i] is None:
            break
        if (
            part.function_target is not None
            or part.jump_target is not None
            or part.is_return
        ):
            break
        base, _ = split_width_suffix(part.mnemonic)
        if base == "mov" and part.args[0] == EBP and part.args[1] == ESP:
            last_frame_op = i
        elif base == "sub" and part.args[0] == ESP:
            last_frame_op = i
        elif base == "push" and is_save_push(i):
            last_frame_op = i
    for i in range(last_frame_op + 1):
        part = body[i]
        if (
            isinstance(part, Instruction)
            and states[i] is not None
            and split_width_suffix(part.mnemonic)[0] == "push"
            and isinstance(part.args[0], Register)
            and not is_save_push(i)
        ):
            alloc_pushes.add(i)

    # Mid-function dead-value pushes: MSVC allocates a call-argument slot with
    # `push <reg>` (one byte, vs three for `sub esp, 4`) and immediately
    # overwrites it, e.g.
    #     push ecx; fstp dword ptr [esp]              (float argument)
    #     push ecx; push ecx; fstp qword ptr [esp]    (double argument)
    # The pushed value never survives, so treat such pushes as pure
    # allocations rather than reads of the (dead) scratch register, which
    # would otherwise become a phantom register argument or an
    # unset-register error.
    def stores_all_of_esp_slot(index: int, slots: int) -> bool:
        """Whether body[index] is an instruction whose only effect on the
        stack is fully overwriting the `slots` dwords at [esp]."""
        part = body[index] if index < len(body) else None
        if not isinstance(part, Instruction) or states[index] is None:
            return False
        base, width = split_width_suffix(part.mnemonic)
        if base not in ("fstp", "fst", "fistp", "fist", "mov"):
            return False
        dest = part.args[0]
        return (
            width == 4 * slots
            and isinstance(dest, AsmAddressMode)
            and dest.base == ESP
            and dest.addend == AsmLiteral(0)
            and not any(isinstance(arg, AsmAddressMode) for arg in part.args[1:])
        )

    for i, part in enumerate(body):
        if (
            i in alloc_pushes
            or not isinstance(part, Instruction)
            or states[i] is None
            or split_width_suffix(part.mnemonic)[0] != "push"
            or not isinstance(part.args[0], Register)
            or part.args[0] == EBP
            or is_save_push(i)
        ):
            continue
        if stores_all_of_esp_slot(i + 1, 1):
            alloc_pushes.add(i)
            continue
        nxt = body[i + 1] if i + 1 < len(body) else None
        if (
            isinstance(nxt, Instruction)
            and states[i + 1] is not None
            and split_width_suffix(nxt.mnemonic)[0] == "push"
            and isinstance(nxt.args[0], Register)
            and not is_save_push(i + 1)
            and stores_all_of_esp_slot(i + 2, 2)
        ):
            alloc_pushes.add(i)
            alloc_pushes.add(i + 1)

    # Pass 3: emit the rewritten body.
    # When ebp is used as a frame pointer, its save/setup/restore
    # (`push ebp; mov ebp, esp; ...; mov esp, ebp; pop ebp` / `leave`) are pure
    # bookkeeping: every `[ebp+k]` access is rewritten against esp, so ebp is
    # never read, and it is left holding its caller value throughout.
    ebp_is_frame_ptr = frame_ebp is not None and not frame_ebp_ambiguous
    new_body: List[BodyPart] = []

    def emit(mnemonic: str, args: List[Argument], meta: InstructionMeta) -> None:
        new_body.append(arch.parse(mnemonic, args, meta.derived()))

    if frame_size > 0:
        first_meta = next(
            (p.meta for p in body if isinstance(p, Instruction)),
            InstructionMeta.missing(),
        )
        emit("sub", [ESP, AsmLiteral(frame_size)], first_meta)

    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            new_body.append(part)
            continue
        st = states[i]
        if st is None:
            # Unreachable code; leave it untouched.
            new_body.append(part)
            continue
        esp, ebp = st
        base, _ = split_width_suffix(part.mnemonic)
        args = part.args

        def adjust(addend: Argument, amount: int) -> Argument:
            if amount == 0:
                return addend
            if isinstance(addend, AsmLiteral):
                return AsmLiteral(addend.value + amount)
            if (
                isinstance(addend, BinOp)
                and addend.op == "+"
                and isinstance(addend.rhs, AsmLiteral)
            ):
                return BinOp("+", addend.lhs, AsmLiteral(addend.rhs.value + amount))
            return BinOp("+", addend, AsmLiteral(amount))

        def rewrite_operand(arg: Argument) -> Argument:
            if not isinstance(arg, AsmAddressMode):
                return arg
            base, addend = arg.base, arg.addend
            if base == ESP:
                return AsmAddressMode(ESP, adjust(addend, frame_size + esp), None)
            if base == EBP and ebp is not None:
                return AsmAddressMode(ESP, adjust(addend, frame_size + ebp), None)
            return arg

        if base == "push":
            loc = frame_size + esp - 4
            if i in alloc_pushes:
                pass  # Pure frame allocation; the value is meaningless.
            elif args[0] == EBP and ebp_is_frame_ptr:
                pass  # Saving the caller's frame pointer; bookkeeping only.
            elif is_save_push(i):
                assert isinstance(args[0], Register)
                emit(
                    "mov",
                    [AsmAddressMode(ESP, AsmLiteral(loc), None), args[0]],
                    part.meta,
                )
            else:
                # An outgoing function call argument.
                emit(
                    "storearg.fictive",
                    [AsmLiteral(loc), rewrite_operand(args[0])],
                    part.meta,
                )
        elif base == "pop":
            if args[0] == EBP and ebp_is_frame_ptr:
                pass  # Restoring the caller's frame pointer; bookkeeping only.
            elif i not in cleanup_pops:
                loc = frame_size + esp
                emit(
                    part.mnemonic.replace("pop", "mov", 1),
                    [
                        rewrite_operand(args[0]),
                        AsmAddressMode(ESP, AsmLiteral(loc), None),
                    ],
                    part.meta,
                )
        elif base in ("pushad", "popad"):
            pass
        elif base in ("sub", "add") and args[0] == ESP:
            pass
        elif base == "and" and args[0] == ESP:
            pass  # Stack alignment; a no-op on the abstracted frame.
        elif base == "mov" and args[0] == ESP:
            pass
        elif base == "mov" and args[0] == EBP and args[1] == ESP and ebp_is_frame_ptr:
            pass  # `mov ebp, esp` frame setup; ebp accesses resolve to esp.
        elif base == "mov" and isinstance(args[0], Register) and args[1] == ESP:
            # Taking the address of the stack into a general register.
            emit(
                "lea",
                [args[0], AsmAddressMode(ESP, AsmLiteral(frame_size + esp), None)],
                part.meta,
            )
        elif base == "lea" and args[0] == ESP:
            pass
        elif base == "leave":
            # `leave` = `mov esp, ebp; pop ebp`; pure frame teardown.
            pass
        elif part.function_target is not None and base == "call":
            # Annotate the call with the base of its stack argument region and
            # the number of argument bytes belonging to it, so translation can
            # split the pending stack arguments across nested calls.
            consume = call_cleanup.get(i, 0)
            if consume == 0:
                consume = cdecl_arg_bytes(i)
            emit(
                "call",
                [
                    rewrite_operand(args[0]),
                    AsmLiteral(frame_size + esp),
                    AsmLiteral(consume),
                ],
                part.meta,
            )
        elif base == "jmp" and isinstance(part.jump_target, JumpTarget):
            if part.jump_target.target in label_pos:
                new_body.append(part)
            else:
                # A terminal jump tears down the current frame. Nothing can
                # follow it, so every pending stack argument belongs to this
                # synthetic call and its argument window is unbounded.
                emit(
                    "call",
                    [
                        AsmGlobalSymbol(part.jump_target.target),
                        AsmLiteral(frame_size + esp),
                        AsmLiteral(-1),
                    ],
                    part.meta,
                )
                emit("ret", [], part.meta)
        else:
            new_args = [rewrite_operand(arg) for arg in args]
            if new_args != args:
                emit(part.mnemonic, new_args, part.meta)
            else:
                new_body.append(part)

    return new_body


# Builders for read-modify-write ALU instructions: (args, old dst value,
# source operand values) -> new dst value.
AluBuilder = Callable[[InstrArgs, Expression, List[Expression]], Expression]
# Builders for single-operand read-modify-write instructions.
UnaryBuilder = Callable[[InstrArgs, Expression], Expression]


# How an instruction affects the flag pseudo-registers:
# - "cmp": full compare-style flags of (dst, src), evaluated *before* the
#   destination is overwritten (like eval_arm_cmp); used by sub and cmp.
#   The c flag is a borrow; see eval_x86_cmp.
# - "add": width-aware flags of an addition result (add/adc), including
#   c = carry-out and hi = (CF==0 && ZF==0).
# - "sbb": subtract-with-borrow flags (sbb): the flags of lhs - (src + CF),
#   computed like a compare so c is a borrow (not logic-op flags).
# - "logic": z/n/hi/ge/gt from the result compared against zero, and
#   c = v = 0 (real x86 semantics for and/or/xor/test; an acceptable
#   approximation for shifts, whose carry-out is rarely consumed).
# - "keep_c": like "logic" but preserving the previous carry flag and
#   setting v from the result (inc/dec semantics); the composite hi predicate
#   is recomputed from the preserved carry plus the new zero flag.
# - "clobber": flags are structurally clobbered but no symbolic value is
#   recorded (rotates, multiplications, divisions).
# - "none": flags are untouched (not/bswap).
class FlagsKind(Enum):
    CMP = "cmp"
    ADD = "add"
    SBB = "sbb"
    LOGIC = "logic"
    KEEP_C = "keep_c"
    CLOBBER = "clobber"
    NONE = "none"


# --- x87 FPU eval helpers (used by X86Arch._parse_fpu) ---
#
# The x87 register stack is eliminated by X86RewritePattern (m2c/x86_fpu.py),
# which rewrites every reachable x87 instruction into a fictive form carrying
# explicit flat virtual registers f0..f7 (e.g. `fadd $f1, $f0`, `fstp.s [m],
# $f2`). The handlers below give those fictive forms their semantics. Virtual
# registers carry Type.floatish() (width-less, like a physical 80-bit slot);
# concrete f32/f64 width only exists at memory boundaries.


def fpu_float_type(width: int) -> Type:
    """The C float type for an x87 memory operand of the given byte width."""
    return Type.f64() if width == 8 else Type.f32()


def fpu_int_type(width: int) -> Type:
    """The signed C integer type for an fild/fistp operand of a given width."""
    return {2: Type.s16(), 4: Type.s32(), 8: Type.s64()}[width]


def load_fild_operand(a: InstrArgs, index: int, width: int) -> Tuple[Expression, Type]:
    """Read an `fild` integer memory operand, returning (value, source int type).

    A 64-bit (`qword`) operand is read as its low and high 4-byte halves rather
    than one signed 8-byte load. MSVC's canonical unsigned u32->float/double
    conversion spills the value with a zero high dword
    (`mov [m], u; xor r,r; mov [m+4], r; fild qword [m]`); a plain signed 8-byte
    load would sign-extend the low dword and corrupt values >= 2^31. A literal-0
    high half means the value is the zero-extended (unsigned) low dword;
    otherwise the halves splice into a genuine signed s64.
    """
    itype = fpu_int_type(width)
    if width != 8:
        return mem_load(a, index, itype), itype
    target = mem_target(a, index)
    hi_target: Union[AddressMode, RawSymbolRef]
    if isinstance(target, AddressMode):
        hi_target = AddressMode(offset=target.offset + 4, base=target.base)
    elif isinstance(target, RawSymbolRef):
        hi_target = RawSymbolRef(offset=target.offset + 4, sym=target.sym)
    else:
        # Scaled-index or otherwise non-decomposable address: signed 8-byte load.
        return mem_load(a, index, itype), itype
    lo = deref(target, a.regs, a.stack_info, size=4)
    hi = deref(hi_target, a.regs, a.stack_info, size=4)
    if early_unwrap(hi) == Literal(0):
        unsigned_lo = as_type(
            as_type(lo, Type.u32(), silent=True), Type.u64(), silent=True
        )
        return unsigned_lo, Type.u64()
    return glue_int64(lo, hi, signed=True), Type.s64()


def is_f64_expr(expr: Expression) -> bool:
    return expr.type.is_float() and expr.type.get_size_bits() == 64


def fpu_binop(
    op: str, lhs: Expression, rhs: Expression, *, reverse: bool = False
) -> Expression:
    """Build an x87 arithmetic binop. Result is f64 when either operand is
    known-f64 (matching C's float->double promotion for the common
    `float op double_constant` pattern), else f32. `reverse` handles the
    fsubr/fdivr forms, which compute `rhs op lhs`."""
    a, b = (rhs, lhs) if reverse else (lhs, rhs)
    if is_f64_expr(lhs) or is_f64_expr(rhs):
        return BinaryOp.f64(a, op, b)
    return BinaryOp.f32(a, op, b)


def f32_literal(value: float) -> Literal:
    """A float constant as an f32-typed Literal holding its IEEE-754 bits."""
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    return Literal(bits, type=Type.f32())


# x87 constant loads. 0/1 render as numeric literals; the transcendental
# constants stay named so matching source can #define them to the exact bits.
FPU_CONSTANTS: Dict[str, Callable[[], Expression]] = {
    "fld1": lambda: f32_literal(1.0),
    "fldz": lambda: f32_literal(0.0),
    "fldpi": lambda: fn_op("M2C_PI", [], Type.f32()),
    "fldl2e": lambda: fn_op("M2C_LOG2E", [], Type.f32()),
    "fldl2t": lambda: fn_op("M2C_LOG2T", [], Type.f32()),
    "fldlg2": lambda: fn_op("M2C_LOG10_2", [], Type.f32()),
    "fldln2": lambda: fn_op("M2C_LN2", [], Type.f32()),
}


def x86_flag_types(width: int) -> Tuple[Type, Type]:
    """(unsigned, signed) types for an x86 operand width in bytes."""
    if width == 1:
        return Type.u8(), Type.s8()
    if width == 2:
        return Type.u16(), Type.s16()
    return Type.u32(), Type.s32()


def eval_x86_cmp(
    s: NodeState, lhs: Expression, rhs: Expression, width: int = 4
) -> None:
    """Set the x86 flag pseudo-registers for `cmp lhs, rhs` (also used by
    sub/neg, which compute the same flags).

    This mirrors evaluate.eval_arm_cmp, with one crucial difference: on x86 the carry
    flag after cmp/sub is a *borrow*, i.e. c = (u32)lhs < (u32)rhs. This is
    the INVERSE of ARM's carry, which is set when there is *no* borrow
    ((u32)lhs >= (u32)rhs). Consumers map jb/jc directly to c, and jae/jnc to
    its negation (see X86Arch.condition_flags)."""
    utype, stype = x86_flag_types(width)
    s.set_reg(Register("z"), BinaryOp.icmp(lhs, "==", rhs))
    sub = BinaryOp.intptr(lhs, "-", rhs)
    sval = as_type(sub, stype, silent=True, unify=False)
    s.set_reg(Register("n"), BinaryOp.scmp(sval, "<", Literal(0)))
    v = fn_op("M2C_OVERFLOW", [sval], Type.boolean())
    s.set_reg(Register("v"), v)
    ulhs = as_type(lhs, utype, silent=True, unify=False)
    slhs = as_type(lhs, stype, silent=True, unify=False)
    urhs = as_type(rhs, utype, silent=True, unify=False)
    srhs = as_type(rhs, stype, silent=True, unify=False)
    s.set_reg(Register("c"), BinaryOp.ucmp(ulhs, "<", urhs))
    s.set_reg(Register("hi"), BinaryOp.ucmp(ulhs, ">", urhs))
    s.set_reg(Register("ge"), BinaryOp.scmp(slhs, ">=", srhs))
    s.set_reg(Register("gt"), BinaryOp.scmp(slhs, ">", srhs))


def fold_literal_add_cmp(cmp: BinaryOp) -> BinaryOp:
    """Fold `(x - c1) == c2` into `x == (c1 + c2)` (and the same for `!=` and
    for `+`). MSVC lowers small switches into in-place `dec reg; je` ladders,
    whose equality flags otherwise render as `(x - 1) == 1` instead of
    `x == 2` (hiding e.g. enum names). Exact for wrapping arithmetic, so only
    applied to equality comparisons."""
    while cmp.op in ("==", "!="):
        lhs, rhs = early_unwrap(cmp.left), early_unwrap(cmp.right)
        if isinstance(lhs, Literal):
            lhs, rhs = rhs, lhs
        if not isinstance(rhs, Literal):
            break
        base, addend = split_imm_addend(lhs)
        if addend == 0:
            break
        cmp = BinaryOp.icmp(base, cmp.op, Literal(rhs.value - addend))
    return cmp


def set_x86_flags_from_result(
    s: NodeState,
    val: Expression,
    width: int = 4,
    *,
    set_c_v: bool = True,
    preserved_carry: Optional[Expression] = None,
) -> None:
    """Set the x86 flag pseudo-registers based on an ALU result, comparing it
    against zero. Used for logic ops (and/or/xor/test/shifts), for which the
    real x86 semantics are CF = OF = 0 (making e.g. `ja` equivalent to
    "result != 0"), and for inc/dec with set_c_v=False (inc/dec preserve the
    carry flag; their overflow flag is set separately).

    For inc/dec, `preserved_carry` supplies the carry flag that the operation
    keeps unchanged, so the composite unsigned-above (ja/jbe) predicate is
    computed as CF==0 && ZF==0 from that preserved carry rather than assuming
    CF==0."""
    _, stype = x86_flag_types(width)
    nez = handle_cmpnez(val)
    assert isinstance(nez, BinaryOp)
    nez = fold_literal_add_cmp(nez)
    s.set_reg(Register("z"), nez.negated())
    sval = as_type(val, stype, silent=True, unify=False)
    s.set_reg(Register("n"), BinaryOp.scmp(sval, "<", Literal(0)))
    if preserved_carry is not None:
        # Unsigned-above (ja) = CF==0 && ZF==0, with CF preserved from before.
        not_carry = condition_from_expr(preserved_carry).negated()
        s.set_reg(
            Register("hi"),
            BinaryOp(left=not_carry, op="&&", right=nez, type=Type.boolean()),
        )
    else:
        # With CF = 0, unsigned-above means "result != 0".
        s.set_reg(Register("hi"), nez)
    s.set_reg(Register("ge"), BinaryOp.scmp(sval, ">=", Literal(0)))
    s.set_reg(Register("gt"), BinaryOp.scmp(sval, ">", Literal(0)))
    if set_c_v:
        s.set_reg(Register("c"), Literal(0))
        s.set_reg(Register("v"), Literal(0))


def set_x86_flags_from_add(
    s: NodeState, lhs: Expression, val: Expression, width: int = 4
) -> None:
    """Set the x86 flag pseudo-registers for an addition (add/adc) whose result
    is `val` and whose left operand is `lhs`, width-aware in `width` bytes.

    Unlike the ARM helper, the carry-out and the composite unsigned predicate
    respect the operand width, so a byte/word add does not get 32-bit carry
    behaviour: CF is the carry-out (the truncated sum wrapped below the left
    operand), and unsigned-above (ja/jbe) is CF==0 && ZF==0."""
    utype, stype = x86_flag_types(width)
    uval = as_type(val, utype, silent=True, unify=False)
    sval = as_type(val, stype, silent=True, unify=False)
    ulhs = as_type(lhs, utype, silent=True, unify=False)
    z = BinaryOp.icmp(val, "==", Literal(0))
    s.set_reg(Register("z"), z)
    s.set_reg(Register("n"), BinaryOp.scmp(sval, "<", Literal(0)))
    # Carry-out: the width-truncated sum wrapped below the left operand.
    carry = BinaryOp.ucmp(uval, "<", ulhs)
    s.set_reg(Register("c"), carry)
    s.set_reg(Register("v"), fn_op("M2C_OVERFLOW", [sval], Type.boolean()))
    # Unsigned-above (ja/jbe): no carry-out and nonzero.
    s.set_reg(
        Register("hi"),
        BinaryOp(left=carry.negated(), op="&&", right=z.negated(), type=Type.boolean()),
    )
    s.set_reg(Register("ge"), BinaryOp.scmp(sval, ">=", Literal(0)))
    s.set_reg(Register("gt"), BinaryOp.scmp(sval, ">", Literal(0)))


# Name of the symbolic x87 status-word marker (fn_op) threaded from an
# fcom-family compare through fnstsw ax to the test-ah idiom. Carries the
# compare's two operands as its arguments.
FNSTSW_MARKER = "M2C_FNSTSW"


def _unwrap_fnstsw_marker(expr: Expression) -> Expression:
    """Remove high-byte extraction and eval-once wrappers around a marker."""
    while True:
        if isinstance(expr, EvalOnceExpr):
            expr = expr.wrapped_expr
        elif isinstance(expr, Cast) and expr.type.is_int():
            expr = expr.expr
        elif (
            isinstance(expr, BinaryOp)
            and expr.op in ("&", ">>")
            and isinstance(expr.right, Literal)
            and (
                (expr.op == "&" and expr.right.value == 0xFF)
                or (expr.op == ">>" and expr.right.value == 8)
            )
        ):
            expr = expr.left
        else:
            return expr


def fnstsw_marker_operands(expr: Expression) -> Optional[Tuple[Expression, Expression]]:
    """If `expr` is an x87 compare status-word marker, its (lhs, rhs) operands;
    otherwise None. Unwraps EvalOnceExpr's *even past a forced/materialized
    boundary* (unlike early_unwrap): a store scheduled between the fcomp and the
    fnstsw/test-ah idiom forces the marker (an fn_op) into a temp via the shared
    store's prevent_later_function_calls, and the fold must still recover the
    operands to reconstruct `x < 0.0f`. The returned operands keep their own
    (possibly forced) wrappers, so a compare operand that genuinely depends on
    the intervening store stays correctly materialized before it."""
    uw = _unwrap_fnstsw_marker(expr)
    if (
        isinstance(uw, FuncCall)
        and isinstance(uw.function, GlobalSymbol)
        and uw.function.symbol_name == FNSTSW_MARKER
        and len(uw.args) == 2
    ):
        return uw.args[0], uw.args[1]
    return None


def fpu_compare_condition(lhs: Expression, rhs: Expression, op: str) -> Condition:
    """A float comparison `lhs op rhs`, as f64 when either operand is
    known-f64 (matching the arithmetic width rule) else f32."""
    if is_f64_expr(lhs) or is_f64_expr(rhs):
        return BinaryOp.dcmp(lhs, op, rhs)
    return BinaryOp.fcmp(lhs, op, rhs)


# TEST AH, mask after FNSTSW AX: the relational operator that is true exactly
# when the x87 compare's ZF would be 1 (i.e. what `jz`/`setz` should read).
# C0 (AH 0x01) = "st0 < src", C3 (AH 0x40) = "st0 == src"; ZF = ((AH & mask)
# == 0). Unordered (NaN) outcomes are folded into the signed direction, as
# MSVC's mask choice implies.
FNSTSW_MASK_OPS: Dict[int, str] = {
    0x01: ">=",  # ZF=1 <=> not(st0 < src)
    0x40: "!=",  # ZF=1 <=> not(st0 == src)
    0x41: ">",  # ZF=1 <=> not(st0 < src) and not(st0 == src)
    0x05: ">=",  # C0|C2: unordered folded in
    0x45: ">",  # C0|C2|C3
    0x44: "!=",  # C2|C3
}


# x87 arithmetic op -> (C operator, reverse-operands flag).
FPU_ARITH_OPS: Dict[str, Tuple[str, bool]] = {
    "fadd": ("+", False),
    "fsub": ("-", False),
    "fsubr": ("-", True),
    "fmul": ("*", False),
    "fdiv": ("/", False),
    "fdivr": ("/", True),
    "faddp": ("+", False),
    "fsubp": ("-", False),
    "fsubrp": ("-", True),
    "fmulp": ("*", False),
    "fdivp": ("/", False),
    "fdivrp": ("/", True),
    "fiadd": ("+", False),
    "fisub": ("-", False),
    "fisubr": ("-", True),
    "fimul": ("*", False),
    "fidiv": ("/", False),
    "fidivr": ("/", True),
}


# x87 unary op -> builder (value -> new value). frndint stays a visible
# intrinsic (its result depends on the rounding-control word, not modeled);
# f2xm1 renders as its libm identity exp2f(x)-1 (its |x|<=1 domain is elided).
def _fpu_is_f32(v: Expression) -> bool:
    """Whether an x87 stack value is pinned to 32-bit float width. x87 registers
    are otherwise width-less floatish (an 80-bit slot); for MSVC's inlined
    fabs/fsqrt -- which are the *double* intrinsics (the float sqrtf/fabsf are
    library calls) and promote their operand -- a width-less operand is f64."""
    return v.type.is_float() and v.type.get_size_bits() == 32


def _fpu_abs(v: Expression) -> Expression:
    # fabs/fsqrt operate on the 80-bit st(0) value; the C intrinsic (and its
    # type) is the f32 form only when that value is a known 32-bit float, else
    # the f64 form (a double, or a promoted/width-less x87 value).
    if _fpu_is_f32(v):
        return fn_op("fabsf", [v], Type.f32())
    return fn_op("fabs", [v], Type.f64())


def _fpu_sqrt(v: Expression) -> Expression:
    if _fpu_is_f32(v):
        return fn_op("sqrtf", [v], Type.f32())
    return fn_op("sqrt", [v], Type.f64())


FPU_UNARY_OPS: Dict[str, Callable[[Expression], Expression]] = {
    "fchs": lambda v: UnaryOp("-", v, type=Type.floatish()),
    "fabs": _fpu_abs,
    "fsqrt": _fpu_sqrt,
    "fsin": lambda v: fn_op("sinf", [v], Type.f32()),
    "fcos": lambda v: fn_op("cosf", [v], Type.f32()),
    "frndint": lambda v: fn_op("M2C_RNDINT", [v], Type.f32()),
    "f2xm1": lambda v: BinaryOp.f32(
        fn_op("exp2f", [v], Type.f32()), "-", f32_literal(1.0)
    ),
    # One-argument x87 CRT math helpers (`fld a; call __CIsin`), taking their
    # f64 argument on and returning their f64 result on the x87 stack (net
    # depth 0). x86_fpu rewrites the `call __CIxxx` to these fictive ops.
    "ci_sqrt.fictive": lambda v: fn_op("sqrt", [v], Type.f64()),
    "ci_sin.fictive": lambda v: fn_op("sin", [v], Type.f64()),
    "ci_cos.fictive": lambda v: fn_op("cos", [v], Type.f64()),
    "ci_tan.fictive": lambda v: fn_op("tan", [v], Type.f64()),
    "ci_exp.fictive": lambda v: fn_op("exp", [v], Type.f64()),
    "ci_log.fictive": lambda v: fn_op("log", [v], Type.f64()),
    "ci_log10.fictive": lambda v: fn_op("log10", [v], Type.f64()),
    "ci_asin.fictive": lambda v: fn_op("asin", [v], Type.f64()),
    "ci_acos.fictive": lambda v: fn_op("acos", [v], Type.f64()),
    "ci_atan.fictive": lambda v: fn_op("atan", [v], Type.f64()),
    "ci_sinh.fictive": lambda v: fn_op("sinh", [v], Type.f64()),
    "ci_cosh.fictive": lambda v: fn_op("cosh", [v], Type.f64()),
    "ci_tanh.fictive": lambda v: fn_op("tanh", [v], Type.f64()),
}

# x87 two-operand transcendentals. Each is `builder(st0, st1) -> value`,
# written into `dst` (0 = st0/top, 1 = st1); `pop` also kills the top. The
# rewrite passes [st0, st1] as the fictive operands.
FPU_BINARY_OPS: Dict[
    str, Tuple[Callable[[Expression, Expression], Expression], int, bool]
] = {
    # fpatan: st1 = atan2(st1, st0), pop.
    "fpatan": (lambda st0, st1: fn_op("atan2f", [st1, st0], Type.f32()), 1, True),
    # fyl2x: st1 = st1 * log2(st0), pop.
    "fyl2x": (
        lambda st0, st1: BinaryOp.f32(st1, "*", fn_op("log2f", [st0], Type.f32())),
        1,
        True,
    ),
    # fyl2xp1: st1 = st1 * log2(st0 + 1), pop.
    "fyl2xp1": (
        lambda st0, st1: BinaryOp.f32(
            st1,
            "*",
            fn_op("log2f", [BinaryOp.f32(st0, "+", f32_literal(1.0))], Type.f32()),
        ),
        1,
        True,
    ),
    # fscale: st0 = st0 * 2**trunc(st1) = ldexpf(st0, (int)st1).
    "fscale": (
        lambda st0, st1: fn_op(
            "ldexpf",
            [st0, handle_convert(st1, Type.s32(), Type.floatish())],
            Type.f32(),
        ),
        0,
        False,
    ),
    # fprem/fprem1: st0 = fmod(st0, st1).
    "fprem": (lambda st0, st1: fn_op("fmodf", [st0, st1], Type.f32()), 0, False),
    "fprem1": (lambda st0, st1: fn_op("fmodf", [st0, st1], Type.f32()), 0, False),
    # Two-argument x87 CRT math helpers (`fld a; fld b; call __CIpow`): the
    # first argument is loaded first (st1), the second last (st0); the helper
    # consumes both and pushes one f64 result in st1's slot (net depth -1).
    # x86_fpu rewrites the `call __CIxxx` to these fictive ops.
    "ci_pow.fictive": (lambda st0, st1: fn_op("pow", [st1, st0], Type.f64()), 1, True),
    "ci_fmod.fictive": (
        lambda st0, st1: fn_op("fmod", [st1, st0], Type.f64()),
        1,
        True,
    ),
    "ci_atan2.fictive": (
        lambda st0, st1: fn_op("atan2", [st1, st0], Type.f64()),
        1,
        True,
    ),
}


class X86HighBytePattern(AsmPattern):
    """Lower reads of ah/bh/ch/dh into explicit full-register extraction."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        part = matcher.input[matcher.index]
        if not isinstance(part, Instruction):
            return None

        written = [reg for reg in part.outputs if reg in HIGH_BYTE_REGS]
        if written:
            name = written[0].register_name
            raise DecompFailure(
                f"writes to high-byte sub-register {name} are not supported yet"
            )

        read = []
        for reg in part.inputs:
            if reg in HIGH_BYTE_REGS and reg not in read:
                read.append(reg)
        if not read:
            return None
        assert len(read) <= 2, "x86 instructions have at most two operands"

        replacements = dict(zip(read, (HI8A, HI8B)))
        new_body: List[ReplacementPart] = [
            AsmInstruction(
                "extract_high_byte.fictive", [replacements[reg], HIGH_BYTE_REGS[reg]]
            )
            for reg in read
        ]
        new_body.append(
            AsmInstruction(
                part.mnemonic,
                [replacements.get(arg, arg) for arg in part.args],
            )
        )
        return Replacement(new_body, 1)


class X86FnstswTestPattern(IrPattern):
    """Give masked byte tests an IR-level x87-status folding opportunity."""

    replacement = "fnstsw_test.fictive v, K"
    parts = ["test.b v, K"]

    def check(self, m: IrMatch, arch: ArchFlowGraph, flow_graph: FlowGraph) -> bool:
        return isinstance(m.symbolic_args["K"], AsmLiteral)


@dataclass(frozen=True)
class X86ContextFacts:
    stdcall_arg_bytes: Mapping[str, int]
    fpu_call_deltas: Mapping[str, int]


EMPTY_X86_CONTEXT_FACTS = X86ContextFacts(MappingProxyType({}), MappingProxyType({}))


def _ctype_is_float(ret_type: CType, typemap: TypeMap) -> bool:
    tp, _ = resolve_typedefs(ret_type, typemap)
    return (
        isinstance(tp, ca.TypeDecl)
        and isinstance(tp.type, ca.IdentifierType)
        and ("float" in tp.type.names or "double" in tp.type.names)
    )


def compute_x86_context_facts(typemap: TypeMap) -> X86ContextFacts:
    stdcall_arg_bytes: Dict[str, int] = {}
    fpu_call_deltas: Dict[str, int] = {}

    def record(mapping: Dict[str, int], name: str, value: int) -> None:
        mapping[name] = value
        if not name.startswith("_"):
            mapping[f"_{name}"] = value

    for name, fn in typemap.functions.items():
        if fn.ret_type is not None and _ctype_is_float(fn.ret_type, typemap):
            record(fpu_call_deltas, name, 1)
        if not fn.is_stdcall or fn.params is None or fn.is_variadic:
            continue
        total = 0
        for param in fn.params:
            tp, _ = resolve_typedefs(param.type, typemap)
            if isinstance(tp, (ca.PtrDecl, ca.ArrayDecl, ca.FuncDecl)):
                size = 4
            else:
                size, _, _ = parse_struct_member(
                    param.type,
                    param.name or "<anonymous>",
                    typemap,
                    allow_unsized=False,
                )
            total += (size + 3) & ~3
        record(stdcall_arg_bytes, name, total)

    facts = X86ContextFacts(
        MappingProxyType(stdcall_arg_bytes), MappingProxyType(fpu_call_deltas)
    )
    return facts


class X86Arch(Arch):
    arch = Target.ArchEnum.X86

    re_comment = r"[#;].*"
    supports_dollar_regs = False
    supports_intel_addressing = True
    # Numbered local code labels emitted by MSVC's COFF tooling.
    re_arch_local_label = re.compile(r"\$L\d+$")

    home_space_size = 0
    base_struct_align = 4

    stack_pointer_reg = ESP
    frame_pointer_regs = [EBP]
    return_address_reg = EIP

    # f0 (the bottom x87 slot) holds a float/double return; not in
    # all_return_regs since x87 values survive calls uncleared (see x86_fpu.py).
    base_return_regs = [(EAX, False), (Register("f0"), True)]
    all_return_regs = [EAX, EDX]
    argument_regs: List[Register] = []
    simple_temp_regs = [ECX, EDX]
    flag_regs = [Register(r) for r in ["n", "z", "c", "v", "hi", "ge", "gt"]]
    # `fsw` carries the symbolic x87 status word between an fcom-family compare
    # and the fnstsw/test-ah idiom that consumes it (see _parse_fpu / eval_cmp).
    # A temp reg so calls clobber it; not a flag reg so it is not swept by the
    # integer-flag machinery.
    fsw_reg = Register("fsw")
    temp_regs = [EAX] + simple_temp_regs + flag_regs + [fsw_reg]
    saved_regs = [EBX, ESI, EDI, EBP, EIP]
    # Raw x87 stack-register names (`st(0)`..`st(7)`, spelled `st0`..`st7`).
    # These reach `parse` only during initial parsing; X86RewritePattern
    # rewrites every reachable use into a flat virtual register before
    # translation, so they never carry real semantics.
    fpu_regs = [Register(f"st{i}") for i in range(8)]
    # The flat x87 virtual registers produced by the FPU prepass (see
    # x86_fpu.py); in neither temp_regs (calls must not clear them) nor
    # saved_regs (no spurious initial "caller value").
    float_regs = [Register(f"f{i}") for i in range(8)]
    # Sub-registers are parsed as their own Register instances so that operand
    # widths survive until normalize_instruction, which rewrites them into
    # full registers plus a width-suffixed mnemonic.
    all_regs = (
        saved_regs
        + temp_regs
        + [stack_pointer_reg]
        + fpu_regs
        + float_regs
        + list(SUB_REGS.keys())
        + list(HIGH_BYTE_REGS.keys())
        + [ZERO]
    )

    aliased_regs: Dict[str, Register] = {}

    asm_patterns: List[AsmPattern] = [
        X86ChkstkPattern(),
        X86PushAllocPattern(),
        X86SehPattern(),
        X86SehEpiloguePattern(),
        X86RawJumpTablePattern(),
        X86RewritePattern(),
        X86HighBytePattern(),
    ]
    ir_patterns = [X86FnstswTestPattern()]

    def return_reg_always_meaningful(self, reg: Register) -> bool:
        # The x87 ABI requires the register stack to be empty at every
        # function boundary except for a float/double return value in st(0).
        # The FPU prepass preserves exactly this property in its flat
        # registers: `f0` is set at a return block iff the x87 stack is
        # non-empty there (see x86_fpu.py). So a value in f0 at every return
        # site *is* the return value in compiler-generated code, no matter
        # how it got there (loaded back from a spill slot, phi'd across
        # branches, or produced by a callee). eax gets no such treatment: it
        # doubles as the primary scratch register, so the generic
        # read-after-write heuristic stays in force for it.
        return reg == Register("f0")

    def missing_return(self) -> List[Instruction]:
        return [self.parse("ret", [], InstructionMeta.missing())]

    def preprocess_instruction(self, mnemonic: str, args: str) -> Tuple[str, str]:
        # String instructions: disassemblers (e.g. capstone) often render the
        # implicit operands explicitly ("rep stosd dword ptr es:[edi], eax").
        # The operands are fixed by the mnemonic, so drop them before the
        # width/segment folding below would mangle the mnemonic. The es:
        # segment marker distinguishes the string form of ambiguous mnemonics
        # (movsd/cmpsd are also SSE2 scalar-double instructions).
        if mnemonic in ("rep", "repe", "repne", "repz", "repnz"):
            parts = args.split(None, 1)
            if parts and parts[0].lower() in STRING_OP_MNEMONICS:
                return mnemonic, parts[0].lower()
        elif mnemonic in STRING_OP_MNEMONICS and "es:" in args.lower():
            return mnemonic, ""

        # Fold "<size> ptr" memory operand prefixes into the mnemonic as a
        # width suffix, and strip syntactic sugar the generic argument parser
        # should not see ("offset symbol" just means the symbol's address,
        # which is how bare symbols are treated anyway).
        widths = [PTR_WIDTHS[m.lower()] for m in RE_PTR.findall(args)]
        explicit_bare_x87_memory = (
            mnemonic in FPU_WIDTHED_MEMORY
            and bool(widths)
            and "[" not in args
            and "," not in args
        )
        args = RE_PTR.sub("", args)
        args = RE_OFFSET.sub("", args)
        args = RE_DISTANCE.sub("", args)
        # Rewrite st(N) FPU registers into parseable names.
        args = RE_ST_REG.sub(lambda m: f"st{m.group(1)}", args)
        # Segment override prefixes. cs/ds/es/ss address the flat default
        # segments in 32-bit code and carry no semantics (some inputs decorate
        # absolute operands with "ds:"), so they are simply stripped. fs/gs
        # genuinely change the address space (on Win32, fs: is the Thread
        # Information Block, e.g. the fs:[0] accesses in SEH prologues), so
        # they move into the mnemonic: the result (e.g. "mov.fs") is either
        # handled explicitly or fails translation with a clear error.
        segments = [m.lower() for m in RE_SEGMENT.findall(args)]
        args = RE_SEGMENT.sub("", args)
        if explicit_bare_x87_memory:
            # Preserve the fact that a `ptr` width was explicit. In particular,
            # dword has no mnemonic suffix, so the normalized address-mode shape
            # distinguishes it from an unsized IDA bare-symbol operand.
            args = f"[{args.strip()}]"
        for seg in segments:
            if seg in ("fs", "gs"):
                mnemonic += f".{seg}"
        if widths:
            # x86 has no instructions with two memory operands of different
            # widths, so all prefixes agree.
            assert len(set(widths)) == 1
            mnemonic += WIDTH_SUFFIXES[widths[0]]
        return mnemonic, args

    def normalize_instruction(
        self, instr: AsmInstruction, asm_state: AsmState
    ) -> AsmInstruction:
        # IDA spells near returns "retn"; it is identical to "ret".
        if instr.mnemonic == "retn":
            instr = AsmInstruction("ret", instr.args)

        # rep/repne/repe prefixes: fold the string instruction into the
        # mnemonic ("rep movsd" -> "rep.movsd").
        if instr.mnemonic in ("rep", "repe", "repne", "repz", "repnz"):
            assert len(instr.args) == 1 and isinstance(instr.args[0], AsmGlobalSymbol)
            op = instr.args[0].symbol_name.lower()
            prefix = {"repz": "repe", "repnz": "repne"}.get(
                instr.mnemonic, instr.mnemonic
            )
            return AsmInstruction(f"{prefix}.{op}", [])

        # Rewrite sub-register operands into full registers, deriving a width
        # suffix from the narrowest sub-register if the mnemonic does not
        # already carry one from a "<size> ptr" prefix.
        sub_width: Optional[int] = None

        def rewrite(arg: Argument) -> Argument:
            nonlocal sub_width
            if isinstance(arg, Register) and arg in SUB_REGS:
                full, width = SUB_REGS[arg]
                if sub_width is None or width < sub_width:
                    sub_width = width
                return full
            if isinstance(arg, Register) and arg in HIGH_BYTE_REGS:
                if sub_width is None or 1 < sub_width:
                    sub_width = 1
                return arg
            if isinstance(arg, AsmAddressMode):
                # Sub-registers cannot appear in 32-bit address modes.
                assert arg.base == ZERO or (
                    arg.base not in SUB_REGS and arg.base not in HIGH_BYTE_REGS
                )
            return arg

        new_args = [rewrite(arg) for arg in instr.args]
        mnemonic = instr.mnemonic
        base, width = split_width_suffix(mnemonic)
        if sub_width is not None and width == 4 and sub_width != 4:
            mnemonic = base + WIDTH_SUFFIXES[sub_width]
        if new_args != instr.args or mnemonic != instr.mnemonic:
            instr = AsmInstruction(mnemonic, new_args)
        return instr

    # Condition code -> (flag register read, negated).
    _flag_z = Register("z")
    _flag_n = Register("n")
    _flag_c = Register("c")
    _flag_v = Register("v")
    _flag_hi = Register("hi")
    _flag_ge = Register("ge")
    _flag_gt = Register("gt")

    condition_flags: Dict[str, Tuple[Register, bool]] = {
        "z": (_flag_z, False),
        "e": (_flag_z, False),
        "nz": (_flag_z, True),
        "ne": (_flag_z, True),
        "s": (_flag_n, False),
        "ns": (_flag_n, True),
        # The c pseudo-register holds x86's carry flag, which after cmp/sub
        # is a *borrow*: c = (u32)lhs < (u32)rhs. (This is the inverse of
        # ARM's carry; see eval_x86_cmp.) jc/jb therefore test c directly.
        "c": (_flag_c, False),
        "b": (_flag_c, False),
        "nae": (_flag_c, False),
        "nc": (_flag_c, True),
        "ae": (_flag_c, True),
        "nb": (_flag_c, True),
        "a": (_flag_hi, False),
        "nbe": (_flag_hi, False),
        "be": (_flag_hi, True),
        "na": (_flag_hi, True),
        "ge": (_flag_ge, False),
        "nl": (_flag_ge, False),
        "l": (_flag_ge, True),
        "nge": (_flag_ge, True),
        "g": (_flag_gt, False),
        "nle": (_flag_gt, False),
        "le": (_flag_gt, True),
        "ng": (_flag_gt, True),
        "o": (_flag_v, False),
        "no": (_flag_v, True),
    }

    # Instruction spec tables, grouped by operand behavior; parse() below
    # interprets these.

    # dst is read and written (register or memory), sources are read.
    # base mnemonic -> (flags behavior, value builder).
    instrs_alu: Dict[str, Tuple[FlagsKind, AluBuilder]] = {
        "add": (FlagsKind.ADD, add_expr),
        "adc": (FlagsKind.ADD, adc_expr),
        "sub": (FlagsKind.CMP, lambda a, l, s: sub_expr(l, s[0])),
        "sbb": (FlagsKind.SBB, sbb_expr),
        "and": (
            FlagsKind.LOGIC,
            lambda a, l, s: replace_bitand(BinaryOp.int(l, "&", s[0])),
        ),
        "or": (FlagsKind.LOGIC, lambda a, l, s: handle_or(l, s[0])),
        "xor": (FlagsKind.LOGIC, lambda a, l, s: BinaryOp.int(l, "^", s[0])),
        "shl": (
            FlagsKind.LOGIC,
            lambda a, l, s: fold_mul_chains(BinaryOp.int(l, "<<", as_intish(s[0]))),
        ),
        "sal": (
            FlagsKind.LOGIC,
            lambda a, l, s: fold_mul_chains(BinaryOp.int(l, "<<", as_intish(s[0]))),
        ),
        "shr": (
            FlagsKind.LOGIC,
            lambda a, l, s: shift_right_expr(l, s[0], signed=False),
        ),
        "sar": (
            FlagsKind.LOGIC,
            lambda a, l, s: shift_right_expr(l, s[0], signed=True),
        ),
        # Rotates only affect the carry/overflow flags on real hardware; we
        # treat the flags as clobbered.
        "rol": (
            FlagsKind.CLOBBER,
            lambda a, l, s: fn_op("ROTATE_LEFT", [l, as_intish(s[0])], Type.intish()),
        ),
        "ror": (
            FlagsKind.CLOBBER,
            lambda a, l, s: fn_op("ROTATE_RIGHT", [l, as_intish(s[0])], Type.intish()),
        ),
        "shrd": (FlagsKind.LOGIC, shrd_expr),
        "shld": (FlagsKind.LOGIC, shld_expr),
    }
    # Shift/rotate instructions, whose count operand may be `cl` (making the
    # width suffix meaningless) and is never sign-extended.
    instrs_shift: Set[str] = {"shl", "sal", "shr", "sar", "rol", "ror", "shrd", "shld"}

    # single operand, read and written.
    instrs_unary: Dict[str, Tuple[FlagsKind, UnaryBuilder]] = {
        "inc": (FlagsKind.KEEP_C, inc_expr),
        "dec": (FlagsKind.KEEP_C, lambda a, v: sub_expr(v, Literal(1))),
        # neg's flags are those of `cmp 0, v`, computed in the eval fn (in
        # particular c = (v != 0), matching x86's CF after neg).
        "neg": (FlagsKind.CMP, lambda a, v: neg_expr(v)),
        "not": (FlagsKind.NONE, lambda a, v: handle_bitinv(v)),
        "bswap": (FlagsKind.NONE, lambda a, v: fn_op("BSWAP32", [v], Type.intish())),
    }

    # dst is written only (not read), src is read; no flags.
    instrs_dst_write: Set[str] = {
        "mov",
        "movsx",
        "movzx",
        "lea",
        "extract_high_byte.fictive",
    }
    # two operands, both read; only flags written.
    instrs_cmp: Set[str] = {"cmp", "test"}
    # instructions with no operands and no structural effects.
    instrs_ignore: Set[str] = {"nop", "int3"}
    # rep-prefixed string instructions: mnemonic -> (inputs, outputs, load, store)
    instrs_string: Dict[str, Tuple[List[Register], List[Register], bool, bool]] = {
        "rep.movsd": ([ESI, EDI, ECX], [ESI, EDI, ECX], True, True),
        "rep.movsw": ([ESI, EDI, ECX], [ESI, EDI, ECX], True, True),
        "rep.movsb": ([ESI, EDI, ECX], [ESI, EDI, ECX], True, True),
        "rep.stosd": ([EDI, EAX, ECX], [EDI, ECX], False, True),
        "rep.stosw": ([EDI, EAX, ECX], [EDI, ECX], False, True),
        "rep.stosb": ([EDI, EAX, ECX], [EDI, ECX], False, True),
        "repne.scasb": ([EDI, ECX, EAX], [EDI, ECX, Register("z")], True, False),
        "repe.cmpsb": ([ESI, EDI, ECX], [ESI, EDI, ECX, Register("z")], True, False),
        "repe.cmpsw": ([ESI, EDI, ECX], [ESI, EDI, ECX, Register("z")], True, False),
        "repe.cmpsd": ([ESI, EDI, ECX], [ESI, EDI, ECX, Register("z")], True, False),
    }
    # Fictive x87 mnemonics produced by X86RewritePattern, dispatched to
    # _parse_fpu. (Raw x87 forms -- which use st(i) names, never f0..f7 --
    # reach `parse` only during initial parsing and fall through to the
    # unknown-instruction handler; the FPU prepass then rewrites them.)
    instrs_fpu: Set[str] = (
        {
            "fld",
            "fild",
            "fld1",
            "fldz",
            "fldpi",
            "fldl2e",
            "fldl2t",
            "fldlg2",
            "fldln2",
            "fmov",
            "fmovpop",
            "fpop",
            "fst",
            "fstp",
            "fistp",
            "fadd",
            "fsub",
            "fsubr",
            "fmul",
            "fdiv",
            "fdivr",
            "faddp",
            "fsubp",
            "fsubrp",
            "fmulp",
            "fdivp",
            "fdivrp",
            "fiadd",
            "fisub",
            "fisubr",
            "fimul",
            "fidiv",
            "fidivr",
            "fchs",
            "fabs",
            "fsqrt",
            "fsin",
            "fcos",
            "fxch",
            "fcom",
            "fcomp",
            "fcompp",
            "fucom",
            "fucomp",
            "fucompp",
            "ftst",
            "ficom",
            "ficomp",
            "frndint",
            "fscale",
            "f2xm1",
            "fprem",
            "fprem1",
            "fpatan",
            "fyl2x",
            "fyl2xp1",
            "fxam",
        }
        | {k for k in FPU_UNARY_OPS if k.startswith("ci_")}
        | {k for k in FPU_BINARY_OPS if k.startswith("ci_")}
    )
    # Non-rep string instructions (single element): mnemonic -> (inputs,
    # outputs, load, store).
    instrs_string_single: Dict[
        str, Tuple[List[Register], List[Register], bool, bool]
    ] = {
        "stosd": ([EDI, EAX], [EDI], False, True),
        "stosw": ([EDI, EAX], [EDI], False, True),
        "stosb": ([EDI, EAX], [EDI], False, True),
        "movsd": ([ESI, EDI], [ESI, EDI], True, True),
        "movsw": ([ESI, EDI], [ESI, EDI], True, True),
        "movsb": ([ESI, EDI], [ESI, EDI], True, True),
    }

    @classmethod
    def _unsupported_eval(
        cls, instr_str: str, reason: str = "no evaluation implemented"
    ) -> Callable[[NodeState, InstrArgs], object]:
        def eval_fn(s: NodeState, a: InstrArgs) -> None:
            raise DecompFailure(f"unsupported x86 instruction ({reason}): {instr_str}")

        return eval_fn

    @classmethod
    def _stack_location(cls, addr: AsmAddressMode) -> Optional[StackLocation]:
        if addr.base == cls.stack_pointer_reg:
            return StackLocation.from_offset(addr.addend)
        return None

    @classmethod
    def _operand_inputs(cls, arg: Argument) -> List[Location]:
        """Locations read in order to evaluate `arg` (for memory operands:
        the registers making up the address, plus the stack location for
        esp-relative addresses, since the operand value is also read)."""
        inputs: List[Location] = []
        for sub in traverse_arg(arg):
            if isinstance(sub, Register) and sub != ZERO and sub not in inputs:
                inputs.append(sub)
        if isinstance(arg, AsmAddressMode):
            stack_loc = cls._stack_location(arg)
            if stack_loc is not None:
                inputs.append(stack_loc)
        return inputs

    @classmethod
    def _address_regs(cls, arg: AsmAddressMode) -> List[Location]:
        """Registers making up a memory operand's address (the operand's
        value itself is not read)."""
        return [
            sub
            for sub in traverse_arg(arg)
            if isinstance(sub, Register) and sub != ZERO
        ]

    @classmethod
    def _flag_outputs(
        cls, flags_kind: FlagsKind
    ) -> Tuple[List[Register], List[Register]]:
        """(outputs, clobbers) among the flag registers for a flags kind."""
        if flags_kind == FlagsKind.NONE:
            return [], []
        if flags_kind == FlagsKind.CLOBBER:
            return [], list(cls.flag_regs)
        if flags_kind == FlagsKind.KEEP_C:
            # inc/dec preserve the carry flag.
            return [r for r in cls.flag_regs if r != cls._flag_c], []
        return list(cls.flag_regs), []

    def parse(
        self, mnemonic: str, args: List[Argument], meta: InstructionMeta
    ) -> Instruction:
        cls = type(self)
        inputs: List[Location] = []
        clobbers: List[Location] = []
        outputs: List[Location] = []
        jump_target: Optional[Union[JumpTarget, Register, List[JumpTarget]]] = None
        function_target: Optional[Argument] = None
        is_conditional = False
        is_return = False
        is_load = False
        is_store = False
        is_effectful = True

        instr_str = str(AsmInstruction(mnemonic, args))
        eval_fn: Optional[Callable[[NodeState, InstrArgs], object]] = (
            cls._unsupported_eval(instr_str)
        )

        base, width = split_width_suffix(mnemonic)

        def add_inputs(arg: Argument) -> None:
            for loc in cls._operand_inputs(arg):
                if loc not in inputs:
                    inputs.append(loc)

        def dest_operand(arg: Argument, *, also_read: bool) -> None:
            """Handle a destination operand (register or memory)."""
            nonlocal is_store
            if isinstance(arg, Register):
                outputs.append(arg)
                if also_read and arg not in inputs:
                    inputs.append(arg)
            elif isinstance(arg, AsmAddressMode):
                is_store = True
                for loc in cls._address_regs(arg):
                    if loc not in inputs:
                        inputs.append(loc)
                stack_loc = cls._stack_location(arg)
                if stack_loc is not None:
                    outputs.append(stack_loc)
                    if also_read and stack_loc not in inputs:
                        inputs.append(stack_loc)
            else:
                raise DecompFailure(f"Invalid x86 destination operand in `{instr_str}`")

        def src_operand(arg: Argument) -> None:
            nonlocal is_load
            if isinstance(arg, AsmAddressMode):
                is_load = True
            add_inputs(arg)

        def write_dst(
            s: NodeState, a: InstrArgs, val: Expression, store_type: Type
        ) -> Optional[Expression]:
            """Write `val` to the first operand. Returns the (wrapped) value
            for register destinations, None for memory destinations."""
            dst = args[0]
            if isinstance(dst, Register):
                return s.set_reg(dst, val)
            src_reg = (
                args[1] if len(args) > 1 and isinstance(args[1], Register) else None
            )
            store = mem_store(a, 0, val, src_reg, store_type)
            if store is not None:
                # The register argument to store_memory is only used for
                # stack spill/restore bookkeeping; fall back to EAX for
                # register-less sources (immediates).
                s.store_memory(store, src_reg if src_reg is not None else EAX)
            return None

        if base == "move.fictive":
            assert len(args) == 2
            dst, src = args
            assert isinstance(dst, Register) and isinstance(src, Register)
            inputs = [src]
            outputs = [dst]
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(dst, a.regs[src])

        elif base == "ret":
            assert len(args) <= 1, "ret takes at most one (immediate) operand"
            inputs = [cls.stack_pointer_reg]
            is_return = True
            eval_fn = None
        elif base == "jmp":
            assert len(args) == 1
            target = args[0]
            if isinstance(target, Register):
                # Indirect jump through a register (computed goto).
                inputs = [target]
                jump_target = target
                is_conditional = True
                eval_fn = lambda s, a: s.set_switch_expr(a.regs[target])
            elif isinstance(target, AsmAddressMode):
                src_operand(target)
                regs = [loc for loc in inputs if isinstance(loc, Register)]
                if regs:
                    # Jump through memory, e.g. `jmp [eax*4 + switchdata]`:
                    # a jump table. Treated like an indirect jump through the
                    # index register; the loaded value drives the switch.
                    jump_target = regs[0]
                    is_conditional = True

                    def eval_fn(s: NodeState, a: InstrArgs) -> None:
                        expr = mem_load(a, 0, Type.reg32(likely_float=False))
                        s.set_switch_expr(expr)

                else:
                    # Register-less jump through an absolute address, e.g.
                    # `jmp [__imp__GetTickCount]`: a tail call through an
                    # import thunk.
                    outputs = list(cls.all_return_regs)
                    clobbers = list(cls.temp_regs)
                    function_target = target
                    is_return = True

                    def eval_fn(s: NodeState, a: InstrArgs) -> None:
                        fn = mem_load(a, 0, Type.reg32(likely_float=False))
                        s.make_function_call(fn, outputs)

            else:
                jump_target = get_jump_target(target)
                eval_fn = None
        elif base.startswith("j") and base[1:] in cls.condition_flags:
            assert len(args) == 1
            flag, negated = cls.condition_flags[base[1:]]
            inputs = [flag]
            jump_target = get_jump_target(args[0])
            is_conditional = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                cond = condition_from_expr(a.regs[flag])
                if negated:
                    cond = cond.negated()
                s.set_branch_condition(cond)

        elif base == "loop":
            assert len(args) == 1
            inputs = [ECX]
            outputs = [ECX]
            jump_target = get_jump_target(args[0])
            is_conditional = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                val = s.set_reg(ECX, sub_expr(a.regs[ECX], Literal(1)))
                s.set_branch_condition(BinaryOp.icmp(val, "!=", Literal(0)))

        elif base == "call":
            # The stack rewrite pass appends two literals: the frame location
            # of [esp] at call time (the base of this call's stack argument
            # region) and the number of argument bytes (-1 if unknown). The FPU
            # prepass may append two more: the x87 virtual register the callee
            # pushes as a float return (fpret, or -1), and the one it consumes
            # off the stack as an argument (fconsume, or -1).
            assert len(args) in (1, 3, 5)
            target = args[0]
            arg_base: Optional[int] = None
            arg_bytes = -1
            fpret = -1
            fconsume = -1
            if len(args) >= 3:
                assert isinstance(args[1], AsmLiteral) and isinstance(
                    args[2], AsmLiteral
                )
                arg_base = args[1].value
                arg_bytes = args[2].value
            if len(args) == 5:
                assert isinstance(args[3], AsmLiteral) and isinstance(
                    args[4], AsmLiteral
                )
                fpret = args[3].value
                fconsume = args[4].value
            inputs = list(cls.argument_regs)
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            function_target = target
            if fpret >= 0:
                # The callee returns its result in st(0), which becomes a new
                # virtual register the call defines. A float/double-returning
                # callee does NOT return anything in eax/edx (they are merely
                # clobbered, via temp_regs), so drop them from the return
                # outputs -- otherwise the return-register heuristic would
                # prefer eax and bit-reinterpret it as the float result.
                outputs = [Register(f"f{fpret}")]
            if fconsume >= 0:
                # The callee consumes st(0) as an argument; it is live into the
                # call (passed as an argument) and killed by it.
                consumed = Register(f"f{fconsume}")
                inputs.append(consumed)
                clobbers.append(consumed)
            math_spec = X86_MATH_HELPERS.get(call_target_symbol(target) or "")
            if math_spec is not None and math_spec.kind == "shift":
                # The shift helpers take their operands in registers (value in
                # edx:eax, count in ecx), so those regs are live into the call.
                for reg in (EAX, EDX, ECX):
                    if reg not in inputs:
                        inputs.append(reg)
            if isinstance(target, Register):
                inputs.append(target)
            elif isinstance(target, AsmAddressMode):
                src_operand(target)
            elif not isinstance(target, (AsmGlobalSymbol, AsmLiteral)):
                raise DecompFailure(f"Invalid x86 call target in `{instr_str}`")

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                if math_spec is not None and eval_math_helper(
                    math_spec, s, a, arg_base
                ):
                    return
                if isinstance(target, Register):
                    fn: Expression = a.regs[target]
                elif isinstance(target, AsmAddressMode):
                    fn = mem_load(a, 0, Type.reg32(likely_float=False))
                else:
                    fn = a.sym_imm(0)
                if fconsume >= 0 and arg_base is not None:
                    # Pass the consumed st(0) value as this call's argument (an
                    # ftol-style helper takes exactly one float in st0). Type it
                    # as the widest float (f64), not floatish: floatish unifies
                    # across call sites to the first concrete width seen, so a
                    # helper called with both a float and a double would narrow
                    # the double argument to f32 and lose precision.
                    consumed = Register(f"f{fconsume}")
                    s.subroutine_args[arg_base] = as_type(
                        a.regs[consumed], Type.f64(), silent=True
                    )
                    del s.regs[consumed]
                if arg_base is None:
                    s.make_function_call(fn, outputs)
                    return
                # Select the pending stack argument stores belonging to this
                # call, and re-key them to the offsets used by function_abi
                # (+4: the ABI is described from the callee's point of view,
                # where [esp] holds the return address).
                leftover = {
                    loc: val
                    for loc, val in s.subroutine_args.items()
                    if loc < arg_base
                    or (arg_bytes >= 0 and loc >= arg_base + arg_bytes)
                }
                selected = {
                    loc - arg_base + 4: val
                    for loc, val in s.subroutine_args.items()
                    if loc not in leftover
                }
                s.subroutine_args.clear()
                s.subroutine_args.update(selected)
                s.make_function_call(fn, outputs)
                # Argument stores for a later call (pushed before this call's
                # arguments) stay pending.
                s.subroutine_args.update(leftover)

        elif base == "storearg.fictive":
            # A rewritten `push` that passes a stack argument to the next
            # `call` (see the stack rewrite pass). args[0] is the frame
            # location of the argument slot.
            assert len(args) == 2 and isinstance(args[0], AsmLiteral)
            arg_loc = args[0].value
            src_operand(args[1])
            outputs = [StackLocation(offset=arg_loc, symbolic_offset=None)]
            is_store = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                s.subroutine_args[arg_loc] = op_value(a, 1, 4)

        elif base == "push":
            assert len(args) == 1
            inputs = [cls.stack_pointer_reg]
            src_operand(args[0])
            outputs = [cls.stack_pointer_reg]
            is_store = True
        elif base == "pop":
            assert len(args) == 1
            inputs = [cls.stack_pointer_reg]
            dest_operand(args[0], also_read=False)
            outputs.append(cls.stack_pointer_reg)
            is_load = True
        elif base == "pushad":
            inputs = [cls.stack_pointer_reg, EAX, ECX, EDX, EBX, EBP, ESI, EDI]
            outputs = [cls.stack_pointer_reg]
            is_store = True
        elif base == "popad":
            inputs = [cls.stack_pointer_reg]
            outputs = [EAX, ECX, EDX, EBX, EBP, ESI, EDI, cls.stack_pointer_reg]
            is_load = True
        elif base in cls.instrs_dst_write:
            assert len(args) == 2
            dst, src = args
            src_operand(src)
            if base == "lea":
                # lea only computes the address; it does not load from it.
                is_load = False
            dest_operand(dst, also_read=False)
            is_effectful = is_store

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                dst, src = args
                if base == "extract_high_byte.fictive":
                    assert isinstance(dst, Register) and isinstance(src, Register)
                    shifted = shift_right_expr(a.regs[src], Literal(8), signed=False)
                    val = replace_bitand(
                        BinaryOp(
                            shifted,
                            "&",
                            Literal(0xFF),
                            type=Type.int_of_size(8),
                        )
                    )
                    s.set_reg(dst, val)
                    return
                if base == "lea":
                    assert isinstance(dst, Register) and isinstance(
                        src, AsmAddressMode
                    ), f"bad lea operands in `{instr_str}`"
                    if src.base != ZERO and isinstance(src.addend, AsmLiteral):
                        # Plain base + offset; for esp-relative addresses this
                        # takes the address of a stack variable.
                        val = handle_addi_real(
                            dst,
                            src.base,
                            a.regs[src.base],
                            Literal(src.addend.value),
                            a,
                        )
                    else:
                        addend = address_expr(src.addend, a)
                        if src.base != ZERO:
                            val = handle_add_real(a.regs[src.base], addend, a)
                        else:
                            val = fold_mul_chains(addend)
                    s.set_reg(dst, val)
                    return
                if base in ("movsx", "movzx"):
                    # The width suffix reflects the (narrower) source operand.
                    assert isinstance(dst, Register)
                    assert width in (1, 2), f"bad {base} source width"
                    if base == "movsx":
                        tp = Type.s8() if width == 1 else Type.s16()
                    else:
                        tp = Type.u8() if width == 1 else Type.u16()
                    if isinstance(src, Register):
                        val = as_type(a.regs[src], tp, silent=False, unify=False)
                    else:
                        val = mem_load(a, 1, tp)
                    s.set_reg(dst, val)
                    return
                if (
                    width < 4
                    and isinstance(dst, Register)
                    and not isinstance(src, AsmLiteral)
                    and _is_zero_value(a.regs.get_raw(dst))
                ):
                    # `xor reg, reg; mov reg_lowN, <mem/reg>`: MSVC's
                    # movzx-equivalent zero-extend idiom. The narrow value is
                    # zero-extended into the just-cleared full register, so it
                    # (and its source) are unsigned -- type it like `movzx`
                    # rather than as a sign-ambiguous partial write, which would
                    # otherwise render an unsigned char/short global as signed.
                    tp = Type.u8() if width == 1 else Type.u16()
                    if isinstance(src, Register):
                        val = as_type(a.regs[src], tp, silent=False, unify=False)
                    else:
                        val = mem_load(a, 1, tp)
                    s.set_reg(dst, val)
                    return
                # mov. Only sign-extend full-width immediates; a byte
                # immediate like `mov cl, 0xff` reads better unsigned.
                val = op_value(a, 1, width, sign_extend_imm=(width == 4))
                if (
                    isinstance(dst, Register)
                    and isinstance(src, Register)
                    and a.stack_info.is_stack_reg(src)
                    and not a.stack_info.is_stack_reg(dst)
                ):
                    # `mov reg, esp`: taking the address of the stack.
                    val = handle_addi_real(dst, src, val, Literal(0), a)
                write_dst(s, a, val, width_type(width))

        elif base in cls.instrs_alu:
            assert len(args) in (2, 3)  # shrd/shld take three operands
            dst = args[0]
            for src in args[1:]:
                src_operand(src)
            dest_operand(dst, also_read=True)
            flags_kind, alu_builder = cls.instrs_alu[base]
            if base in ("adc", "sbb"):
                inputs.append(cls._flag_c)
            flag_outs, flag_clobbers = cls._flag_outputs(flags_kind)
            outputs.extend(flag_outs)
            clobbers.extend(flag_clobbers)
            is_effectful = is_store

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                w = width
                if base in cls.instrs_shift and isinstance(args[-1], Register):
                    # The width suffix came from a `cl` shift count; the
                    # shift itself operates on the full destination.
                    w = 4
                if (
                    base in ("add", "sub")
                    and isinstance(args[0], Register)
                    and a.stack_info.is_stack_reg(args[0])
                    and isinstance(args[1], AsmLiteral)
                ):
                    # Stack pointer adjustment (prologue/epilogue). The frame
                    # size is tracked by get_stack_info; leave esp alone.
                    return
                if base == "xor" and args[0] == args[1]:
                    # xor r, r: the idiom for zeroing a register.
                    assert isinstance(args[0], Register)
                    zero = s.set_reg(args[0], Literal(0))
                    set_x86_flags_from_result(s, zero, w)
                    return
                if (
                    base == "sbb"
                    and args[0] == args[1]
                    and isinstance(args[0], Register)
                ):
                    # sbb r, r: materializes the borrow as 0 / -1 (see
                    # sbb_expr). The result is independent of r's prior value,
                    # so don't read the operands at all -- r is often dead
                    # here, and reading it would register a phantom argument
                    # (or an unset-register error). The flags of `sbb r, r`
                    # match those of `cmp 0, borrow`: c' = borrow (0 - 1
                    # borrows), z = !borrow, n = borrow (result is -1), no
                    # signed overflow.
                    borrow = carry_in(a)
                    val = s.set_reg(args[0], UnaryOp("-", borrow, type=Type.intish()))
                    eval_x86_cmp(s, Literal(0), as_intish(borrow), w)
                    return
                if (
                    w == 4
                    and isinstance(args[0], Register)
                    and isinstance(args[1], AsmLiteral)
                ):
                    # `or reg, -1` / `and reg, 0`: constant-result idioms
                    # (MSVC's compact `mov reg, -1`). Don't read reg -- it is
                    # often dead here, and reading it would yield a spurious
                    # M2C_ERROR(unset register).
                    imm = args[1].value & 0xFFFFFFFF
                    const: Optional[int] = None
                    if base == "or" and imm == 0xFFFFFFFF:
                        const = -1
                    elif base == "and" and imm == 0:
                        const = 0
                    if const is not None:
                        val = s.set_reg(args[0], Literal(const))
                        set_x86_flags_from_result(s, val, w)
                        return
                lhs = op_value(a, 0, w)
                sign_ext = base not in cls.instrs_shift
                srcs = [
                    op_value(a, i, w, sign_extend_imm=sign_ext)
                    for i in range(1, len(args))
                ]
                # Compute the value before writing flags: adc/sbb read the
                # incoming carry, which the FlagsKind.SBB write below overwrites.
                val = alu_builder(a, lhs, srcs)

                if flags_kind == FlagsKind.CMP:
                    # Compare-style flags are based on the values *before*
                    # the destination is overwritten.
                    eval_x86_cmp(s, lhs, srcs[0], w)
                elif flags_kind == FlagsKind.SBB:
                    # sbb: subtract-with-borrow flags = flags of
                    # lhs - (src + carry-in), a compare (c is a borrow), also
                    # taken before the destination is overwritten.
                    eval_x86_cmp(s, lhs, BinaryOp.intptr(srcs[0], "+", carry_in(a)), w)

                def set_alu_flags(result: Expression) -> None:
                    if flags_kind == FlagsKind.ADD:
                        set_x86_flags_from_add(s, lhs, result, w)
                    elif flags_kind == FlagsKind.LOGIC:
                        set_x86_flags_from_result(s, result, w)

                if isinstance(args[0], Register):
                    val = s.set_reg(args[0], val)
                    set_alu_flags(val)
                else:
                    # For memory destinations, set flags before the store so
                    # that flag expressions refer to pre-store values.
                    set_alu_flags(val)
                    write_dst(s, a, val, width_type(w))

        elif base in cls.instrs_unary:
            assert len(args) == 1
            dest_operand(args[0], also_read=True)
            if isinstance(args[0], AsmAddressMode):
                is_load = True
            elif isinstance(args[0], Register) and args[0] not in inputs:
                inputs.append(args[0])
            flags_kind, unary_builder = cls.instrs_unary[base]
            if flags_kind == FlagsKind.KEEP_C:
                # inc/dec preserve the carry flag but fold it into the composite
                # unsigned-above predicate (ja/jbe), so they read it.
                inputs.append(cls._flag_c)
            flag_outs, flag_clobbers = cls._flag_outputs(flags_kind)
            outputs.extend(flag_outs)
            clobbers.extend(flag_clobbers)
            is_effectful = is_store

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                old = op_value(a, 0, width)
                if flags_kind == FlagsKind.CMP:
                    # neg: flags of `cmp 0, old` (c = borrow = (old != 0)).
                    eval_x86_cmp(s, Literal(0), old, width)
                # inc/dec keep CF; fold the preserved carry into the composite
                # unsigned-above predicate (read before it can be overwritten).
                keep_carry = (
                    a.regs[cls._flag_c] if flags_kind == FlagsKind.KEEP_C else None
                )

                def set_unary_flags(result: Expression) -> None:
                    if flags_kind == FlagsKind.KEEP_C:
                        set_x86_flags_from_result(
                            s, result, width, set_c_v=False, preserved_carry=keep_carry
                        )
                        s.set_reg(
                            cls._flag_v,
                            fn_op("M2C_OVERFLOW", [result], Type.boolean()),
                        )

                val = unary_builder(a, old)
                if isinstance(args[0], Register):
                    val = s.set_reg(args[0], val)
                    set_unary_flags(val)
                else:
                    set_unary_flags(val)
                    write_dst(s, a, val, width_type(width))

        elif base == "fnstsw_test.fictive":
            assert len(args) == 2
            status_reg, mask = args
            assert isinstance(status_reg, Register)
            assert isinstance(mask, (AsmLiteral, AsmGlobalSymbol))
            inputs = [status_reg]
            outputs = list(cls.flag_regs)
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                assert isinstance(mask, AsmLiteral)
                status = a.regs[status_reg]
                operands = fnstsw_marker_operands(status)
                op = FNSTSW_MASK_OPS.get(mask.value & 0xFF)
                if operands is not None and op is not None:
                    cond = fpu_compare_condition(operands[0], operands[1], op)
                    for flag in cls.flag_regs:
                        s.set_reg(flag, cond)
                    return
                high = shift_right_expr(status, Literal(8), signed=False)
                value = replace_bitand(BinaryOp.int(high, "&", Literal(mask.value)))
                set_x86_flags_from_result(s, value, 1)

        elif base in cls.instrs_cmp:
            assert len(args) == 2
            for arg in args:
                src_operand(arg)
            outputs = list(cls.flag_regs)
            is_effectful = False

            def eval_cmp(s: NodeState, a: InstrArgs) -> None:
                lhs = op_value(a, 0, width)
                if base == "test":
                    # CF = OF = 0; other flags are based on lhs & rhs. The
                    # dominant idiom `test r, r` just compares r against 0.
                    if args[0] == args[1]:
                        val = lhs
                    else:
                        rhs = op_value(a, 1, width)
                        val = replace_bitand(BinaryOp.int(lhs, "&", rhs))
                    set_x86_flags_from_result(s, val, width)
                else:
                    rhs = op_value(a, 1, width)
                    eval_x86_cmp(s, lhs, rhs, width)

            eval_fn = eval_cmp
        elif base.startswith("set") and base[3:] in cls.condition_flags:
            assert len(args) == 1
            flag, negated = cls.condition_flags[base[3:]]
            inputs = [flag]
            dest_operand(args[0], also_read=False)
            is_effectful = is_store

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                cond = condition_from_expr(a.regs[flag])
                if negated:
                    cond = cond.negated()
                # setcc writes a 0/1 byte; for register destinations this is
                # modeled as writing the full register (usually zeroed
                # beforehand by `xor r, r`).
                val = Cast(expr=cond, reinterpret=False, silent=True, type=Type.u8())
                write_dst(s, a, val, Type.int_of_size(8))

        elif base == "cdq":
            assert not args
            inputs = [EAX]
            outputs = [EDX]
            is_effectful = False
            eval_fn = lambda s, a: s.set_reg(
                EDX, BinaryOp.sint(a.regs[EAX], ">>", Literal(31))
            )
        elif base == "rdtsc":
            assert not args
            outputs = [EAX, EDX]
            clobbers = list(cls.flag_regs)
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                val = fn_op("M2C_RDTSC", [], Type.u64())
                s.set_reg(EAX, val)
                s.set_reg(
                    EDX, fn_op("SECOND_REG", [val], Type.reg32(likely_float=False))
                )

        elif base in cls.instrs_string_single and not args:
            # Non-rep string instructions perform a single element operation
            # and advance esi/edi (assuming the direction flag is clear).
            str_inputs, str_outputs, is_load, is_store = cls.instrs_string_single[base]
            inputs = list(str_inputs)
            outputs = list(str_outputs)
            elem = {"b": 1, "w": 2, "d": 4}[base[-1]]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                op = base[:-1]
                tp = width_type(elem)
                if op in ("stos", "movs"):
                    if op == "stos":
                        value: Expression = a.regs[EAX]
                        if elem < 4:
                            value = as_type(value, tp, silent=True, unify=False)
                    else:
                        value = deref(a.regs[ESI], a.regs, a.stack_info, size=elem)
                    dest = deref(
                        a.regs[EDI], a.regs, a.stack_info, size=elem, store=True
                    )
                    dest.type.unify(tp)
                    s.write_statement(
                        StoreStmt(source=as_type(value, tp, silent=False), dest=dest)
                    )
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", Literal(elem)))
                    if op == "movs":
                        s.set_reg(ESI, BinaryOp.intptr(a.regs[ESI], "+", Literal(elem)))
                else:
                    raise DecompFailure(f"x86 `{instr_str}` is not supported")

        elif base in ("mul", "imul", "div", "idiv") and len(args) <= 1:
            # One-operand forms operate on edx:eax.
            inputs = [EAX] if base in ("mul", "imul") else [EAX, EDX]
            if args:
                src_operand(args[0])
            outputs = [EAX, EDX]
            clobbers = list(cls.flag_regs)
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                if not args:
                    raise DecompFailure(f"x86 `{instr_str}` is missing its operand")
                src = op_value(a, 0, width)
                acc = a.regs[EAX]
                if base in ("mul", "imul"):
                    hi_op = "MULT_HI" if base == "imul" else "MULTU_HI"
                    s.set_reg(EAX, BinaryOp.int(acc, "*", src))
                    s.set_reg(EDX, fold_divmod(BinaryOp.int(acc, hi_op, src)))
                else:
                    # The 64-bit dividend edx:eax is assumed to be the
                    # sign/zero-extension of eax (set up via cdq or
                    # `xor edx, edx`), which is how compilers emit 32-bit
                    # division; edx's value is not consulted.
                    if base == "idiv":
                        quot = BinaryOp.sint(acc, "/", src)
                        rem = BinaryOp.sint(acc, "%", src)
                    else:
                        quot = BinaryOp.uint(acc, "/", src)
                        rem = BinaryOp.uint(acc, "%", src)
                    s.set_reg(EAX, quot)
                    s.set_reg(EDX, rem)

        elif base == "imul":
            # Two/three-operand forms only write the destination register.
            assert len(args) in (2, 3) and isinstance(args[0], Register)
            inputs = [args[0]] if len(args) == 2 else []
            for src in args[1:]:
                src_operand(src)
            outputs = [args[0]]
            clobbers = list(cls.flag_regs)
            is_effectful = False

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                assert isinstance(args[0], Register)
                if len(args) == 2:
                    lhs: Expression = a.regs[args[0]]
                    rhs = op_value(a, 1, width)
                else:
                    lhs = op_value(a, 1, width)
                    rhs = op_value(a, 2, width)
                s.set_reg(args[0], fold_mul_chains(BinaryOp.int(lhs, "*", rhs)))

        elif base == "xchg":
            assert len(args) == 2
            for arg in args:
                src_operand(arg)
            for arg in args:
                dest_operand(arg, also_read=True)
            is_effectful = is_store

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                vals = [op_value(a, 0, width), op_value(a, 1, width)]
                for i in (0, 1):
                    dst = args[i]
                    val = vals[1 - i]
                    if isinstance(dst, Register):
                        s.set_reg(dst, val)
                    else:
                        other = args[1 - i]
                        src_reg = other if isinstance(other, Register) else None
                        store = mem_store(a, i, val, src_reg, width_type(width))
                        if store is not None:
                            s.store_memory(
                                s, store, src_reg if src_reg is not None else EAX
                            )

        elif base == "mov.fs" and len(args) == 2:
            # An fs-segment absolute access (on Win32, the Thread Information
            # Block). The canonical SEH-chain bookkeeping at fs:[0] is
            # removed by X86SehPattern before translation; any other access
            # (e.g. the stack bounds at fs:[4]/fs:[8]) is modeled as an
            # explicit opaque platform read/write.
            fs_dst, fs_src = args
            if (
                isinstance(fs_dst, Register)
                and isinstance(fs_src, AsmAddressMode)
                and fs_src.base == ZERO
                and isinstance(fs_src.addend, AsmLiteral)
            ):
                offset = fs_src.addend.value
                outputs = [fs_dst]
                is_load = True

                def eval_fn(s: NodeState, a: InstrArgs) -> None:
                    assert isinstance(fs_dst, Register)
                    s.set_reg(
                        fs_dst,
                        fn_op("M2C_FS_LOAD", [Literal(offset)], width_type(width)),
                    )

            elif (
                isinstance(fs_dst, AsmAddressMode)
                and fs_dst.base == ZERO
                and isinstance(fs_dst.addend, AsmLiteral)
            ):
                store_offset = fs_dst.addend.value
                src_operand(fs_src)
                is_store = True

                def eval_fn(s: NodeState, a: InstrArgs) -> None:
                    value = op_value(a, 1, width)
                    s.write_statement(
                        void_fn_op("M2C_FS_STORE", [Literal(store_offset), value])
                    )

        elif mnemonic in cls.instrs_string:
            str_inputs, str_outputs, is_load, is_store = cls.instrs_string[mnemonic]
            inputs = list(str_inputs)
            outputs = list(str_outputs)
            elem_size = {"b": 1, "w": 2, "d": 4}[mnemonic[-1]]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                count = as_intish(a.regs[ECX])
                if elem_size != 1:
                    count = fold_mul_chains(
                        BinaryOp.int(count, "*", Literal(elem_size))
                    )
                op = mnemonic.split(".")[1][:-1] if "." in mnemonic else ""
                if op == "movs":
                    # rep movsX: copy ecx elements from [esi] to [edi].
                    s.write_statement(
                        void_fn_op("M2C_MEMCPY", [a.regs[EDI], a.regs[ESI], count])
                    )
                    s.set_reg(ESI, BinaryOp.intptr(a.regs[ESI], "+", count))
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", count))
                    s.set_reg(ECX, Literal(0))
                elif op == "stos":
                    # rep stosX: fill ecx elements at [edi] with al/ax/eax.
                    value = a.regs[EAX]
                    if elem_size < 4:
                        value = as_type(
                            value, width_type(elem_size), silent=True, unify=False
                        )
                    fn_name = {1: "M2C_MEMSET", 2: "M2C_MEMSET16", 4: "M2C_MEMSET32"}[
                        elem_size
                    ]
                    s.write_statement(
                        void_fn_op(
                            fn_name, [a.regs[EDI], value, as_intish(a.regs[ECX])]
                        )
                    )
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", count))
                    s.set_reg(ECX, Literal(0))
                elif op == "scas":
                    # repne scasb: scan [edi...] for the byte in al, leaving
                    # edi one past the match and decrementing ecx once per
                    # byte scanned. With al = 0 this is the strlen idiom; for
                    # other (or statically unknown) al values, model the scan
                    # as a memchr. Both assume the byte is found before ecx
                    # runs out, which holds for the compiler-emitted
                    # strlen/strcpy expansions this models.
                    al = early_unwrap(a.regs[EAX])
                    if isinstance(al, Literal) and al.value & 0xFF == 0:
                        length: Expression = fn_op(
                            "M2C_STRLEN", [a.regs[EDI]], Type.u32()
                        )
                    else:
                        needle = as_type(
                            a.regs[EAX], Type.u8(), silent=True, unify=False
                        )
                        found = fn_op(
                            "M2C_MEMCHR",
                            [a.regs[EDI], needle, as_intish(a.regs[ECX])],
                            Type.ptr(),
                        )
                        length = BinaryOp.intptr(found, "-", a.regs[EDI])
                    advance = BinaryOp.int(length, "+", Literal(1))
                    s.set_reg(ECX, BinaryOp.int(a.regs[ECX], "-", advance))
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", advance))
                    s.set_reg(Register("z"), Literal(1, type=Type.boolean()))
                else:
                    # repe cmpsX: memcmp-style comparison; only the z flag
                    # outcome is modeled.
                    assert op == "cmps"
                    cmp = fn_op(
                        "M2C_MEMCMP", [a.regs[ESI], a.regs[EDI], count], Type.s32()
                    )
                    s.set_reg(ESI, BinaryOp.intptr(a.regs[ESI], "+", count))
                    s.set_reg(EDI, BinaryOp.intptr(a.regs[EDI], "+", count))
                    s.set_reg(ECX, Literal(0))
                    s.set_reg(Register("z"), BinaryOp.icmp(cmp, "==", Literal(0)))

        elif base in cls.instrs_ignore:
            is_effectful = False
            eval_fn = None
        elif base in ("fnstsw", "fstsw", "fldcw", "fstcw", "fnstcw") or (
            base in cls.instrs_fpu
            and any(isinstance(a, Register) and a.is_float() for a in args)
        ):
            # A fictive x87 instruction emitted by X86RewritePattern (raw
            # x87 forms name st(i), never f0..f7, so the f-register gate
            # excludes them). The status/control-word ops have no f-register
            # operand, so they are dispatched by name.
            return self._parse_fpu(base, width, args, meta, instr_str)
        else:
            # Unknown instruction (x87 FPU, etc.). Guess a structural shape
            # so that file parsing and flow graph construction still work;
            # evaluation will fail with a clear error.
            eval_fn = cls._unsupported_eval(instr_str, "unknown instruction")
            for arg in args:
                add_inputs(arg)
            if args and isinstance(args[0], Register):
                inputs = [loc for loc in inputs if loc != args[0]]
                outputs = [args[0]]

        surviving_high_bytes = [
            arg for arg in args if isinstance(arg, Register) and arg in HIGH_BYTE_REGS
        ]
        if surviving_high_bytes:
            name = surviving_high_bytes[0].register_name

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                raise DecompFailure(
                    f"read from high-byte sub-register {name} survived lowering"
                )

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

    def _parse_fpu(
        self,
        base: str,
        width: int,
        args: List[Argument],
        meta: InstructionMeta,
        instr_str: str,
    ) -> Instruction:
        """Build an Instruction for a fictive x87 op (see the FPU eval helpers
        above and m2c/x86_fpu.py). Virtual registers f0..f7 are ordinary
        registers to the rest of m2c; the only novelty is that popping ops
        *kill* their consumed register (drop it from outputs, list it in
        clobbers, and `del s.regs[...]` in eval) so a popped value stops
        existing -- which is what makes float-return detection sound."""
        cls = type(self)
        inputs: List[Location] = []
        outputs: List[Location] = []
        clobbers: List[Location] = []
        is_load = False
        is_store = False
        eval_fn: Optional[Callable[[NodeState, InstrArgs], object]]

        def add_operand_inputs(arg: Argument) -> None:
            for loc in cls._operand_inputs(arg):
                if loc not in inputs:
                    inputs.append(loc)

        def not_implemented(reason: str) -> Callable[[NodeState, InstrArgs], None]:
            for arg in args:
                add_operand_inputs(arg)

            def fail(s: NodeState, a: InstrArgs) -> None:
                raise DecompFailure(
                    f"unsupported x87 instruction ({reason}): {instr_str}"
                )

            return fail

        # --- Loads: dst register + memory source ---
        if base in ("fld", "fild"):
            dst = args[0]
            assert isinstance(dst, Register)
            add_operand_inputs(args[1])
            outputs = [dst]
            is_load = True
            if base == "fld":
                ftype = fpu_float_type(width)

                def eval_fn(s: NodeState, a: InstrArgs) -> None:
                    s.set_reg(dst, mem_load(a, 1, ftype))

            else:

                def eval_fn(s: NodeState, a: InstrArgs) -> None:
                    val, itype = load_fild_operand(a, 1, width)
                    s.set_reg(dst, handle_convert(val, Type.floatish(), itype))

        # --- Constants: 0/1 as numeric literals, pi/log constants as named
        # macros so matching source can #define them. ---
        elif base in FPU_CONSTANTS:
            dst = args[0]
            assert isinstance(dst, Register)
            outputs = [dst]
            make_const = FPU_CONSTANTS[base]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(dst, make_const())

        # --- Register moves (fld/fst st(i), fstp st(i) with i>0) ---
        elif base in ("fmov", "fmovpop"):
            dst, src = args
            assert isinstance(dst, Register) and isinstance(src, Register)
            inputs = [src]
            outputs = [dst]
            pop = base == "fmovpop"
            if pop:
                clobbers = [src]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(dst, a.regs[src])
                if pop:
                    del s.regs[src]

        # --- Pop-discard (fstp st(0)) ---
        elif base == "fpop":
            reg = args[0]
            assert isinstance(reg, Register)
            inputs = [reg]
            clobbers = [reg]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                del s.regs[reg]

        # --- Stores (fst/fstp to memory) ---
        elif base in ("fst", "fstp"):
            src = args[1]
            assert isinstance(src, Register)
            add_operand_inputs(args[0])
            if src not in inputs:
                inputs.append(src)
            stack_loc = (
                cls._stack_location(args[0])
                if isinstance(args[0], AsmAddressMode)
                else None
            )
            if stack_loc is not None:
                outputs.append(stack_loc)
            is_store = True
            pop = base == "fstp"
            if pop:
                clobbers = [src]
            ftype = fpu_float_type(width)

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                store = mem_store(a, 0, a.regs[src], src, ftype)
                if store is not None:
                    s.store_memory(store, src)
                if pop:
                    del s.regs[src]

        # --- Non-popping arithmetic: dst register op src (register or mem) ---
        elif base in ("fadd", "fsub", "fsubr", "fmul", "fdiv", "fdivr"):
            dst, src = args
            assert isinstance(dst, Register)
            inputs = [dst]
            if isinstance(src, Register):
                if src not in inputs:
                    inputs.append(src)
            else:
                add_operand_inputs(src)
                is_load = True
            outputs = [dst]
            op, reverse = FPU_ARITH_OPS[base]
            ftype = fpu_float_type(width)

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                lhs = a.regs[dst]
                rhs = (
                    a.regs[src] if isinstance(src, Register) else mem_load(a, 1, ftype)
                )
                s.set_reg(dst, fpu_binop(op, lhs, rhs, reverse=reverse))

        # --- Popping arithmetic (faddp st(i), st): dst op st0, then pop st0 ---
        elif base in ("faddp", "fsubp", "fsubrp", "fmulp", "fdivp", "fdivrp"):
            dst, src = args
            assert isinstance(dst, Register) and isinstance(src, Register)
            inputs = [dst, src]
            outputs = [dst]
            clobbers = [src]
            op, reverse = FPU_ARITH_OPS[base]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                lhs = a.regs[dst]
                rhs = a.regs[src]
                s.set_reg(dst, fpu_binop(op, lhs, rhs, reverse=reverse))
                del s.regs[src]

        # --- Unary operations on the top of stack ---
        elif base in FPU_UNARY_OPS:
            reg = args[0]
            assert isinstance(reg, Register)
            inputs = [reg]
            outputs = [reg]
            builder = FPU_UNARY_OPS[base]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                s.set_reg(reg, builder(a.regs[reg]))

        # --- fxch: swap two slots ---
        elif base == "fxch":
            ra, rb = args
            assert isinstance(ra, Register) and isinstance(rb, Register)
            inputs = [ra, rb]
            outputs = [ra, rb]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                va = a.regs[ra]
                vb = a.regs[rb]
                s.set_reg(ra, vb)
                s.set_reg(rb, va)

        # --- Compares: store a symbolic status-word marker into `fsw`, killing
        # any popped operands. The fnstsw/test-ah idiom below consumes it. ---
        elif base in (
            "fcom",
            "fcomp",
            "fucom",
            "fucomp",
            "fcompp",
            "fucompp",
            "ftst",
            "ficom",
            "ficomp",
        ):
            top = args[0]
            assert isinstance(top, Register)
            inputs = [top]
            popped: List[Register] = []
            if base in ("fcomp", "fucomp", "ficomp"):
                popped = [top]
            elif base in ("fcompp", "fucompp"):
                assert isinstance(args[1], Register)
                popped = [top, args[1]]
            clobbers = list(popped)
            outputs = [cls.fsw_reg]
            rhs_arg = args[1] if len(args) > 1 else None
            if isinstance(rhs_arg, Register):
                if rhs_arg not in inputs:
                    inputs.append(rhs_arg)
            elif rhs_arg is not None:
                add_operand_inputs(rhs_arg)
                is_load = True
            is_int_cmp = base in ("ficom", "ficomp")
            itype = fpu_int_type(width) if is_int_cmp else None

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                lhs = a.regs[top]
                if base == "ftst":
                    rhs: Expression = f32_literal(0.0)
                elif is_int_cmp:
                    assert itype is not None
                    rhs = handle_convert(mem_load(a, 1, itype), Type.floatish(), itype)
                elif isinstance(rhs_arg, Register):
                    rhs = a.regs[rhs_arg]
                else:
                    rhs = mem_load(a, 1, fpu_float_type(width))
                operands = (lhs, rhs)
                s.set_reg(
                    cls.fsw_reg,
                    fn_op(
                        FNSTSW_MARKER,
                        list(operands),
                        Type.u16(),
                        marker=True,
                    ),
                )
                for reg in popped:
                    del s.regs[reg]

        # --- fnstsw ax: move the status-word marker into eax for the test. ---
        elif base in ("fnstsw", "fstsw"):
            assert isinstance(args[0], Register)
            eax = args[0]
            inputs = [cls.fsw_reg]
            outputs = [eax]

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                if cls.fsw_reg in s.regs:
                    s.set_reg(eax, s.regs[cls.fsw_reg])
                else:
                    # A stray fnstsw with no preceding compare: surface it.
                    s.set_reg(eax, fn_op(FNSTSW_MARKER, [], Type.u16()))

        # --- fistp: store the top as an integer (truncating cast), then pop.
        # The rounding mode is assumed fixed globally, so a C truncation cast
        # matches the ambient chop mode. ---
        elif base == "fistp":
            src = args[1]
            assert isinstance(src, Register)
            add_operand_inputs(args[0])
            if src not in inputs:
                inputs.append(src)
            stack_loc = (
                cls._stack_location(args[0])
                if isinstance(args[0], AsmAddressMode)
                else None
            )
            if stack_loc is not None:
                outputs.append(stack_loc)
            is_store = True
            clobbers = [src]
            itype = fpu_int_type(width)

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                casted = handle_convert(a.regs[src], itype, Type.floatish())
                store = mem_store(a, 0, casted, None, itype)
                if store is not None:
                    s.store_memory(store, src)
                del s.regs[src]

        # --- Integer-operand arithmetic: top op (float)int_load ---
        elif base in ("fiadd", "fisub", "fisubr", "fimul", "fidiv", "fidivr"):
            dst = args[0]
            assert isinstance(dst, Register)
            inputs = [dst]
            add_operand_inputs(args[1])
            outputs = [dst]
            is_load = True
            op, reverse = FPU_ARITH_OPS[base]
            itype = fpu_int_type(width)

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                lhs = a.regs[dst]
                rhs = handle_convert(mem_load(a, 1, itype), Type.floatish(), itype)
                s.set_reg(dst, fpu_binop(op, lhs, rhs, reverse=reverse))

        # --- Control word: kept as visible intrinsics. The rounding/precision
        # mode is not modeled, so surface the load/store so a human sees the
        # mode changes rather than pretending they vanish. ---
        elif base == "fldcw":
            add_operand_inputs(args[0])
            is_load = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                s.write_statement(void_fn_op("M2C_FLDCW", [mem_load(a, 0, Type.u16())]))

        elif base in ("fstcw", "fnstcw"):
            add_operand_inputs(args[0])
            stack_loc = (
                cls._stack_location(args[0])
                if isinstance(args[0], AsmAddressMode)
                else None
            )
            if stack_loc is not None:
                outputs.append(stack_loc)
            is_store = True

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                store = mem_store(
                    a, 0, fn_op("M2C_FSTCW", [], Type.u16()), None, Type.u16()
                )
                if store is not None:
                    s.store_memory(store, EAX)

        # --- Two-operand transcendentals (fpatan/fyl2x/fscale/fprem/...) ---
        elif base in FPU_BINARY_OPS:
            st0, st1 = args
            assert isinstance(st0, Register) and isinstance(st1, Register)
            builder, dst_idx, pop = FPU_BINARY_OPS[base]
            dst = st1 if dst_idx == 1 else st0
            inputs = [st0, st1] if st0 != st1 else [st0]
            outputs = [dst]
            if pop:
                clobbers = [st0]  # the top is consumed

            def eval_fn(s: NodeState, a: InstrArgs) -> None:
                val = builder(a.regs[st0], a.regs[st1])
                s.set_reg(dst, val)
                if pop:
                    del s.regs[st0]

        # --- Anything left really is unhandled: fail cleanly. ---
        else:
            eval_fn = not_implemented("x87 op")

        return Instruction(
            mnemonic=base if width == 4 else base + WIDTH_SUFFIXES[width],
            args=args,
            meta=meta,
            inputs=inputs,
            clobbers=clobbers,
            outputs=outputs,
            jump_target=None,
            function_target=None,
            is_conditional=False,
            is_return=False,
            is_store=is_store,
            is_load=is_load,
            is_effectful=is_store,
            eval_fn=eval_fn,
        )

    def default_function_abi_candidate_slots(self) -> List[AbiArgSlot]:
        return []

    def arg_name(self, loc: ArgLoc) -> str:
        if loc.offset is not None:
            if loc.offset >= 4 and loc.offset % 4 == 0:
                return f"arg{(loc.offset - 4) // 4}"
            return f"arg_{loc.offset:x}"
        assert loc.reg is not None
        return loc.reg.register_name

    def function_abi(
        self,
        fn_sig: FunctionSignature,
        likely_regs: Dict[Register, bool],
        *,
        for_call: bool,
    ) -> Abi:
        """cdecl: all arguments are passed on the stack, pushed right to left.
        Offsets are relative to the stack pointer at function entry; slot 0
        (the return address) is skipped, so the first argument is at +4."""
        known_slots: List[AbiArgSlot] = []
        offset = 4
        if fn_sig.params_known:
            for i, param in enumerate(fn_sig.params):
                param_type = param.type.decay()
                size, align = param_type.get_parameter_size_align_bytes()
                size = (size + 3) & ~3
                # 32-bit x86 (cdecl/stdcall) passes every stack argument in
                # 4-byte slots regardless of the value's natural alignment: an
                # 8-byte `double` still occupies two consecutive slots starting
                # at a 4-byte boundary (a leading `double` is at [esp+4], not
                # [esp+8]). Cap the alignment at 4 while keeping the 8-byte
                # size, so `double`/`long long` arguments land at the right
                # offsets. (get_parameter_size_align_bytes returns (8, 8) for
                # scalars, which is the host/64-bit ABI's alignment.)
                align = min(align, 4)
                offset = (offset + align - 1) & -align
                known_slots.append(
                    AbiArgSlot(
                        ArgLoc(offset, i, None),
                        param_type,
                        name=param.name,
                    )
                )
                offset += size
        candidate_slots: List[AbiArgSlot] = []
        # For calls, unknown extra arguments are collected from the pending
        # stack argument stores instead (make_function_call requires
        # possible_slots to be registers).
        if not for_call and (not fn_sig.params_known or fn_sig.is_variadic):
            # __fastcall (and __thiscall's `this`) passes the first
            # register-sized arguments in ecx/edx. Offer them as candidates:
            # compiler-generated cdecl/stdcall code never reads these
            # caller-save registers before setting them, so this only fires
            # for functions that really do take register arguments, which
            # would otherwise decode as unset-register errors.
            candidate_slots.append(AbiArgSlot(ArgLoc(None, 0, ECX), Type.any_reg()))
            candidate_slots.append(AbiArgSlot(ArgLoc(None, 1, EDX), Type.any_reg()))
            for i in range(8):
                candidate_slots.append(
                    AbiArgSlot(
                        ArgLoc(offset + 4 * i, len(known_slots) + 2 + i, None),
                        Type.any_reg(),
                    )
                )
        return Abi(arg_slots=known_slots, possible_slots=candidate_slots)

    def function_return(self, expr: Expression) -> Dict[Register, Expression]:
        # Return values are in eax (edx holds the high half of u64's) or, for
        # float/double returns, in x87 st(0). A call that returns a float
        # pushes it onto the FPU stack; the FPU prepass annotates the call with
        # exactly which virtual register that is (fpret), so every f0..f7 needs
        # an entry here even though only one is ever an output of a given call.
        result = {
            EAX: as_type(expr, Type.intptr(), silent=True, unify=False),
            EDX: fn_op("SECOND_REG", [expr], Type.reg32(likely_float=False)),
        }
        for i in range(8):
            result[Register(f"f{i}")] = Cast(
                expr, reinterpret=True, silent=True, type=Type.floatish()
            )
        return result
