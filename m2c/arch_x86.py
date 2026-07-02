"""i386 (x86) architecture support, for Ghidra-exported Intel-syntax asm.

Phase 1 established registration, parsing, and structural instruction
information (inputs/outputs/jump targets). Phase 2a adds real semantics
(eval_fns) for data operations, flags, and conditionals: mov/movsx/movzx/lea/
xchg, the ALU (incl. mul/imul/div/idiv/cdq and shifts), cmp/test, all jcc and
setcc, and loads/stores through all addressing modes. Stack modeling for
push/pop, function call arguments, jump tables/switches, and x87 are still
unimplemented (phase 2b); those instructions parse structurally but raise
DecompFailure during translation.

Design notes:

- Operand widths: x86 encodes operand sizes both in memory operand prefixes
  ("byte ptr"/"word ptr"/"dword ptr"/"qword ptr") and in sub-register names
  (al/ah/ax/...). Both are canonicalized into a mnemonic suffix during
  parsing/normalization, ARM-style: `mov.b`, `mov.w`, `mov.q`. The default
  32-bit width has no suffix ("dword ptr" and plain 32-bit registers map to
  a bare mnemonic). This happens in two places:
    * `preprocess_instruction` strips "<size> ptr" from the argument string
      (before the generic argument parser runs) and appends the suffix;
    * `normalize_instruction` rewrites sub-register operands (al -> eax etc.)
      and appends the suffix derived from the narrowest sub-register, if the
      mnemonic doesn't already carry one.
  This keeps width information available in `X86Arch.parse` (and the eval
  functions) without extending the shared Argument types.

- Sub-register writes: an instruction that writes a sub-register (`mov.b` to
  cl, setcc, a byte load, ...) is deliberately modeled as writing the *full*
  storage register with a partial-width-typed value. This is correct for the
  overwhelmingly common patterns (`xor reg, reg` + byte load, setcc after
  clearing the register, byte loads that feed byte stores) but loses the
  upper bits of the old register value in the rare cases where they are
  live across the sub-register write.

- Memory operands: Intel bracket expressions are parsed by the shared parser
  (gated by the `supports_intel_addressing` capability) into AsmAddressMode,
  with an Optional base register: `[esp + 0xc]` -> base=esp, addend=0xc;
  `[symbol]` -> base=None, addend=symbol; scaled indices stay in the addend
  as BinOp trees: `[esi + ebx*8 + 0x30]` -> base=esi,
  addend=(ebx * 8) + 0x30. During evaluation these become either an
  AddressMode (base + literal offset; handles stack accesses), a RawSymbolRef
  (absolute [symbol + off]), or a generic address Expression whose shape
  (base + index * scale + offset) `deref`/`array_access_from_add` recognize
  and turn into array indexing.

- Flags: mirrors ARM's condition flag scheme (z, n, c, v plus the composite
  hi/ge/gt pseudo-registers): flag-setting instructions store complete
  symbolic conditions into the flag registers, and consumers (jcc/setcc/
  adc/sbb) read them back, negating via Condition.negated() where needed.
  One crucial difference from ARM: after `cmp a, b` (or sub/neg), x86's
  carry flag is a *borrow*: c = (u32)a < (u32)b, the inverse of ARM's carry.
  After additions, c is the carry-out (same as ARM). See eval_x86_cmp in
  evaluate.py and `condition_flags` below.

- Arguments: cdecl passes all arguments on the stack. At function entry
  [esp + 0] holds the return address, so with an unmoved stack pointer the
  arguments live at [esp + 4], [esp + 8], ... A literal `sub esp, N` in the
  prologue is folded into StackInfo.allocated_stack_size (see
  get_stack_info), moving the argument region to [esp + N + 4]. Arbitrary
  mid-function esp adjustment (push/pop in particular) is not modeled yet.
"""

from __future__ import annotations
import re
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from .error import DecompFailure
from .options import Target
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
    get_jump_target,
    traverse_arg,
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
    Cast,
    Expression,
    InstrArgs,
    Literal,
    NodeState,
    RawSymbolRef,
    RegExpression,
    StoreStmt,
    Type,
    UnaryOp,
    as_intish,
    as_type,
    parse_symbol_ref,
)
from .evaluate import (
    condition_from_expr,
    deref,
    eval_x86_cmp,
    fn_op,
    fold_divmod,
    fold_mul_chains,
    handle_add_real,
    handle_addi_real,
    handle_bitinv,
    handle_or,
    handle_sub,
    make_store_real,
    replace_bitand,
    set_arm_flags_from_add,
    set_x86_flags_from_result,
    shift_right_expr,
)
from .types import FunctionSignature


EAX = Register("eax")
ECX = Register("ecx")
EDX = Register("edx")
EBX = Register("ebx")
ESP = Register("esp")
EBP = Register("ebp")
ESI = Register("esi")
EDI = Register("edi")
EIP = Register("eip")

# Sub-register name -> (full register, width in bytes)
SUB_REGS: Dict[Register, Tuple[Register, int]] = {
    Register("al"): (EAX, 1),
    Register("ah"): (EAX, 1),
    Register("ax"): (EAX, 2),
    Register("bl"): (EBX, 1),
    Register("bh"): (EBX, 1),
    Register("bx"): (EBX, 2),
    Register("cl"): (ECX, 1),
    Register("ch"): (ECX, 1),
    Register("cx"): (ECX, 2),
    Register("dl"): (EDX, 1),
    Register("dh"): (EDX, 1),
    Register("dx"): (EDX, 2),
    Register("si"): (ESI, 2),
    Register("di"): (EDI, 2),
    Register("bp"): (EBP, 2),
    Register("sp"): (ESP, 2),
}

WIDTH_SUFFIXES: Dict[int, str] = {1: ".b", 2: ".w", 4: "", 8: ".q"}
PTR_WIDTHS: Dict[str, int] = {"byte": 1, "word": 2, "dword": 4, "qword": 8}

RE_PTR = re.compile(r"\b(byte|word|dword|qword)\s+ptr\s+", re.IGNORECASE)
RE_OFFSET = re.compile(r"\boffset\s+", re.IGNORECASE)
RE_ST_REG = re.compile(r"\bst\((\d)\)", re.IGNORECASE)
RE_SEGMENT = re.compile(r"\b([cdefgs]s):", re.IGNORECASE)


def split_width_suffix(mnemonic: str) -> Tuple[str, int]:
    """Split e.g. "mov.b" into ("mov", 1). No suffix means 4 bytes."""
    for width, suffix in WIDTH_SUFFIXES.items():
        if suffix and mnemonic.endswith(suffix):
            return mnemonic[: -len(suffix)], width
    return mnemonic, 4


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


def mem_target(a: InstrArgs, index: int) -> Union[AddressMode, RawSymbolRef, Expression]:
    """Compute the target of a memory operand, as either an AddressMode
    (base register + literal offset, which also handles esp-relative stack
    accesses), a RawSymbolRef (absolute [symbol + offset]), or a generic
    address Expression (scaled-index modes, which deref turns into array
    accesses)."""
    arg = a.raw_arg(index)
    assert isinstance(arg, AsmAddressMode), f"expected a memory operand, found {arg}"
    if arg.base is not None and isinstance(arg.addend, AsmLiteral):
        return AddressMode(offset=arg.addend.value, base=arg.base)
    if arg.base is None:
        ref = parse_symbol_ref(arg.addend)
        if ref is not None:
            return ref
    addend = address_expr(arg.addend, a)
    if arg.base is None:
        return addend
    return BinaryOp.intptr(a.regs[arg.base], "+", addend)


def mem_load(a: InstrArgs, index: int, type: Type) -> Expression:
    size = type.get_size_bytes()
    assert size is not None
    target = mem_target(a, index)
    expr = deref(target, a.regs, a.stack_info, size=size)
    return as_type(expr, type, silent=True)


def mem_store(
    a: InstrArgs,
    index: int,
    value: Expression,
    value_reg: Optional[Register],
    type: Type,
) -> Optional[StoreStmt]:
    size = type.get_size_bytes()
    assert size is not None
    target = mem_target(a, index)
    source_raw: Optional[RegExpression] = None
    if value_reg is not None:
        source_raw = a.regs.get_raw(value_reg)
    if isinstance(target, (AddressMode, RawSymbolRef)):
        return make_store_real(value, source_raw, target, a.regs, a.stack_info, type)
    dest = deref(target, a.regs, a.stack_info, size=size, store=True)
    dest.type.unify(type)
    return StoreStmt(source=as_type(value, type, silent=False), dest=dest)


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
        if width < 4:
            # Reading a sub-register: reinterpret the low bits.
            val = as_type(val, type or width_type(width), silent=True, unify=False)
        return val
    if isinstance(arg, AsmAddressMode):
        return mem_load(a, index, type or width_type(width))
    imm = a.full_imm(index)
    if isinstance(imm, Literal) and sign_extend_imm:
        return sign_extended_imm(imm.value, width)
    return imm


def sub_expr(lhs: Expression, rhs: Expression) -> Expression:
    val = handle_sub(lhs, rhs)
    if isinstance(val, BinaryOp):
        val = fold_divmod(val)
    return fold_mul_chains(val)


def carry_in(a: InstrArgs) -> Expression:
    """The x86 carry flag as a 0/1 integer expression (for adc/sbb)."""
    return condition_from_expr(a.regs[Register("c")])


def adc_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    return handle_add_real(handle_add_real(lhs, srcs[0], a), carry_in(a), a)


def sbb_expr(a: InstrArgs, lhs: Expression, srcs: List[Expression]) -> Expression:
    if a.raw_arg(0) == a.raw_arg(1):
        # sbb r, r: idiom for materializing the carry (borrow) flag as
        # 0 / -1 without branching.
        return UnaryOp("-", carry_in(a), type=Type.intish())
    return BinaryOp.intptr(handle_sub(lhs, srcs[0]), "-", carry_in(a))


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


# Builders for read-modify-write ALU instructions: (args, old dst value,
# source operand values) -> new dst value.
AluBuilder = Callable[[InstrArgs, Expression, List[Expression]], Expression]
# Builders for single-operand read-modify-write instructions.
UnaryBuilder = Callable[[InstrArgs, Expression], Expression]

# How an instruction affects the flag pseudo-registers:
# - "cmp": full compare-style flags of (dst, src), evaluated *before* the
#   destination is overwritten (like eval_arm_cmp); used by sub and cmp.
#   The c flag is a borrow; see eval_x86_cmp.
# - "add": flags of an addition result, including c = carry-out.
# - "logic": z/n/hi/ge/gt from the result compared against zero, and
#   c = v = 0 (real x86 semantics for and/or/xor/test; an acceptable
#   approximation for shifts, whose carry-out is rarely consumed).
# - "keep_c": like "logic" but preserving the previous carry flag and
#   setting v from the result (inc/dec semantics).
# - "clobber": flags are structurally clobbered but no symbolic value is
#   recorded (rotates, multiplications, divisions).
# - "none": flags are untouched (not/bswap).
FLAGS_CMP = "cmp"
FLAGS_ADD = "add"
FLAGS_LOGIC = "logic"
FLAGS_KEEP_C = "keep_c"
FLAGS_CLOBBER = "clobber"
FLAGS_NONE = "none"


class X86Arch(Arch):
    arch = Target.ArchEnum.X86

    re_comment = r"[#;].*"
    supports_dollar_regs = False
    supports_intel_addressing = True

    home_space_size = 0
    base_struct_align = 4

    stack_pointer_reg = ESP
    frame_pointer_regs = [EBP]
    return_address_reg = EIP

    base_return_regs = [(EAX, False)]
    all_return_regs = [EAX, EDX]
    argument_regs: List[Register] = []
    simple_temp_regs = [ECX, EDX]
    flag_regs = [Register(r) for r in ["n", "z", "c", "v", "hi", "ge", "gt"]]
    temp_regs = [EAX] + simple_temp_regs + flag_regs
    saved_regs = [EBX, ESI, EDI, EBP, EIP]
    # x87 FPU stack registers (untranslated, but must parse)
    fpu_regs = [Register(f"st{i}") for i in range(8)]
    # Sub-registers are parsed as their own Register instances so that operand
    # widths survive until normalize_instruction, which rewrites them into
    # full registers plus a width-suffixed mnemonic.
    all_regs = (
        saved_regs + temp_regs + [stack_pointer_reg] + fpu_regs + list(SUB_REGS.keys())
    )

    aliased_regs: Dict[str, Register] = {}

    def missing_return(self) -> List[Instruction]:
        return [self.parse("ret", [], InstructionMeta.missing())]

    def preprocess_instruction(self, mnemonic: str, args: str) -> Tuple[str, str]:
        # Fold "<size> ptr" memory operand prefixes into the mnemonic as a
        # width suffix, and strip syntactic sugar the generic argument parser
        # should not see ("offset symbol" just means the symbol's address,
        # which is how bare symbols are treated anyway).
        widths = [PTR_WIDTHS[m.lower()] for m in RE_PTR.findall(args)]
        args = RE_PTR.sub("", args)
        args = RE_OFFSET.sub("", args)
        # Rewrite st(N) FPU registers into parseable names.
        args = RE_ST_REG.sub(lambda m: f"st{m.group(1)}", args)
        # Segment override prefixes (e.g. the fs:[0] accesses in SEH
        # prologues): move the segment into the mnemonic. The resulting
        # mnemonic (e.g. "mov.fs") is treated as an unknown instruction, which
        # parses fine structurally but fails translation with a clear error.
        segments = [m.lower() for m in RE_SEGMENT.findall(args)]
        args = RE_SEGMENT.sub("", args)
        for seg in segments:
            mnemonic += f".{seg}"
        if widths:
            # x86 has no instructions with two memory operands of different
            # widths, so all prefixes agree.
            mnemonic += WIDTH_SUFFIXES[min(widths)]
        return mnemonic, args

    def normalize_instruction(
        self, instr: AsmInstruction, asm_state: AsmState
    ) -> AsmInstruction:
        # rep/repne/repe prefixes: fold the string instruction into the
        # mnemonic ("rep movsd" -> "rep.movsd").
        if instr.mnemonic in ("rep", "repe", "repne", "repz", "repnz"):
            assert len(instr.args) == 1 and isinstance(instr.args[0], AsmGlobalSymbol)
            op = instr.args[0].symbol_name.lower()
            return AsmInstruction(f"{instr.mnemonic}.{op}", [])

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
            if isinstance(arg, AsmAddressMode):
                # (Sub-registers cannot appear in 32-bit address modes;
                # this is just defensive.)
                base = arg.base
                if base is not None and base in SUB_REGS:
                    base = SUB_REGS[base][0]
                return AsmAddressMode(base, arg.addend, arg.writeback)
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
    instrs_alu: Dict[str, Tuple[str, AluBuilder]] = {
        "add": (FLAGS_ADD, lambda a, l, s: handle_add_real(l, s[0], a)),
        "adc": (FLAGS_ADD, adc_expr),
        "sub": (FLAGS_CMP, lambda a, l, s: sub_expr(l, s[0])),
        "sbb": (FLAGS_LOGIC, sbb_expr),
        "and": (FLAGS_LOGIC, lambda a, l, s: replace_bitand(BinaryOp.int(l, "&", s[0]))),
        "or": (FLAGS_LOGIC, lambda a, l, s: handle_or(l, s[0])),
        "xor": (FLAGS_LOGIC, lambda a, l, s: BinaryOp.int(l, "^", s[0])),
        "shl": (
            FLAGS_LOGIC,
            lambda a, l, s: fold_mul_chains(BinaryOp.int(l, "<<", as_intish(s[0]))),
        ),
        "sal": (
            FLAGS_LOGIC,
            lambda a, l, s: fold_mul_chains(BinaryOp.int(l, "<<", as_intish(s[0]))),
        ),
        "shr": (FLAGS_LOGIC, lambda a, l, s: shift_right_expr(l, s[0], signed=False)),
        "sar": (FLAGS_LOGIC, lambda a, l, s: shift_right_expr(l, s[0], signed=True)),
        # Rotates only affect the carry/overflow flags on real hardware; we
        # treat the flags as clobbered.
        "rol": (
            FLAGS_CLOBBER,
            lambda a, l, s: fn_op("ROTATE_LEFT", [l, as_intish(s[0])], Type.intish()),
        ),
        "ror": (
            FLAGS_CLOBBER,
            lambda a, l, s: fn_op("ROTATE_RIGHT", [l, as_intish(s[0])], Type.intish()),
        ),
        "shrd": (FLAGS_LOGIC, shrd_expr),
        "shld": (FLAGS_LOGIC, shld_expr),
    }
    # Shift/rotate instructions, whose count operand may be `cl` (making the
    # width suffix meaningless) and is never sign-extended.
    instrs_shift: Set[str] = {"shl", "sal", "shr", "sar", "rol", "ror", "shrd", "shld"}

    # single operand, read and written.
    instrs_unary: Dict[str, Tuple[str, UnaryBuilder]] = {
        "inc": (FLAGS_KEEP_C, lambda a, v: handle_add_real(v, Literal(1), a)),
        "dec": (FLAGS_KEEP_C, lambda a, v: sub_expr(v, Literal(1))),
        # neg's flags are those of `cmp 0, v`, computed in the eval fn (in
        # particular c = (v != 0), matching x86's CF after neg).
        "neg": (FLAGS_CMP, lambda a, v: UnaryOp.sint("-", v)),
        "not": (FLAGS_NONE, lambda a, v: handle_bitinv(v)),
        "bswap": (FLAGS_NONE, lambda a, v: fn_op("BSWAP32", [v], Type.intish())),
    }

    # dst is written only (not read), src is read; no flags.
    instrs_dst_write: Set[str] = {"mov", "movsx", "movzx", "lea"}
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
    }

    @classmethod
    def _unsupported_eval(
        cls, instr_str: str, reason: str = "phase 2b"
    ) -> Callable[[NodeState, InstrArgs], object]:
        def eval_fn(s: NodeState, a: InstrArgs) -> None:
            raise DecompFailure(
                f"x86 instruction evaluation is not implemented yet ({reason}): {instr_str}"
            )

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
            if isinstance(sub, Register) and sub not in inputs:
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
        return [sub for sub in traverse_arg(arg) if isinstance(sub, Register)]

    @classmethod
    def _flag_outputs(
        cls, flags_kind: str
    ) -> Tuple[List[Register], List[Register]]:
        """(outputs, clobbers) among the flag registers for a flags kind."""
        if flags_kind == FLAGS_NONE:
            return [], []
        if flags_kind == FLAGS_CLOBBER:
            return [], list(cls.flag_regs)
        if flags_kind == FLAGS_KEEP_C:
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
            src_reg = args[1] if len(args) > 1 and isinstance(args[1], Register) else None
            store = mem_store(a, 0, val, src_reg, store_type)
            if store is not None:
                # The register argument to store_memory is only used for
                # stack spill/restore bookkeeping; fall back to EAX for
                # register-less sources (immediates).
                s.store_memory(store, src_reg if src_reg is not None else EAX)
            return None

        if base == "ret":
            assert len(args) <= 1, "ret takes at most one (immediate) operand"
            inputs = [cls.stack_pointer_reg]
            is_return = True
            eval_fn = None
        elif base == "jmp":
            assert len(args) == 1
            target = args[0]
            if isinstance(target, Register):
                # Indirect jump (jump table); target resolution is phase 2b.
                inputs = [target]
                jump_target = target
                is_conditional = True
                eval_fn = cls._unsupported_eval(instr_str, "jump tables")
            elif isinstance(target, AsmAddressMode):
                src_operand(target)
                regs = [loc for loc in inputs if isinstance(loc, Register)]
                if regs:
                    # Jump through memory, e.g. `jmp [eax*4 + switchdata]`.
                    # Treat like an indirect jump through the index register.
                    jump_target = regs[0]
                    is_conditional = True
                    eval_fn = cls._unsupported_eval(instr_str, "jump tables")
                else:
                    # Register-less jump through an absolute address, e.g.
                    # `jmp [__imp__GetTickCount]`: a tail call through an
                    # import thunk.
                    outputs = list(cls.all_return_regs)
                    function_target = target
                    is_return = True
            else:
                jump_target = get_jump_target(target)
                eval_fn = None
        elif base.startswith("j") and base[1:] in cls.condition_flags:
            assert len(args) == 1
            flag, negated = cls.condition_flags[base[1:]]
            inputs = [flag]
            jump_target = get_jump_target(args[0])
            is_conditional = True

            def eval_jcc(s: NodeState, a: InstrArgs) -> None:
                cond = condition_from_expr(a.regs[flag])
                if negated:
                    cond = cond.negated()
                s.set_branch_condition(cond)

            eval_fn = eval_jcc
        elif base == "loop":
            assert len(args) == 1
            inputs = [ECX]
            outputs = [ECX]
            jump_target = get_jump_target(args[0])
            is_conditional = True
        elif base == "call":
            assert len(args) == 1
            target = args[0]
            inputs = list(cls.argument_regs)
            outputs = list(cls.all_return_regs)
            clobbers = list(cls.temp_regs)
            function_target = target
            if isinstance(target, Register):
                inputs.append(target)
            elif isinstance(target, AsmAddressMode):
                src_operand(target)
            elif not isinstance(target, (AsmGlobalSymbol, AsmLiteral)):
                raise DecompFailure(f"Invalid x86 call target in `{instr_str}`")
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

            def eval_dst_write(s: NodeState, a: InstrArgs) -> None:
                dst, src = args
                if base == "lea":
                    assert isinstance(dst, Register) and isinstance(
                        src, AsmAddressMode
                    ), f"bad lea operands in `{instr_str}`"
                    if src.base is not None and isinstance(src.addend, AsmLiteral):
                        # Plain base + offset; for esp-relative addresses this
                        # takes the address of a stack variable.
                        val = handle_addi_real(
                            dst, src.base, a.regs[src.base], Literal(src.addend.value), a
                        )
                    else:
                        addend = address_expr(src.addend, a)
                        if src.base is not None:
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

            eval_fn = eval_dst_write
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

            def eval_alu(s: NodeState, a: InstrArgs) -> None:
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
                lhs = op_value(a, 0, w)
                sign_ext = base not in cls.instrs_shift
                srcs = [
                    op_value(a, i, w, sign_extend_imm=sign_ext)
                    for i in range(1, len(args))
                ]
                if flags_kind == FLAGS_CMP:
                    # Compare-style flags are based on the values *before*
                    # the destination is overwritten.
                    eval_x86_cmp(s, lhs, srcs[0], w)
                val = alu_builder(a, lhs, srcs)
                if isinstance(args[0], Register):
                    val = s.set_reg(args[0], val)
                    if flags_kind == FLAGS_ADD:
                        set_arm_flags_from_add(s, val)
                    elif flags_kind == FLAGS_LOGIC:
                        set_x86_flags_from_result(s, val, w)
                else:
                    # For memory destinations, set flags before the store so
                    # that flag expressions refer to pre-store values.
                    if flags_kind == FLAGS_ADD:
                        set_arm_flags_from_add(s, val)
                    elif flags_kind == FLAGS_LOGIC:
                        set_x86_flags_from_result(s, val, w)
                    write_dst(s, a, val, width_type(w))

            eval_fn = eval_alu
        elif base in cls.instrs_unary:
            assert len(args) == 1
            dest_operand(args[0], also_read=True)
            if isinstance(args[0], AsmAddressMode):
                is_load = True
            elif isinstance(args[0], Register) and args[0] not in inputs:
                inputs.append(args[0])
            flags_kind, unary_builder = cls.instrs_unary[base]
            flag_outs, flag_clobbers = cls._flag_outputs(flags_kind)
            outputs.extend(flag_outs)
            clobbers.extend(flag_clobbers)
            is_effectful = is_store

            def eval_unary(s: NodeState, a: InstrArgs) -> None:
                old = op_value(a, 0, width)
                if flags_kind == FLAGS_CMP:
                    # neg: flags of `cmp 0, old` (c = borrow = (old != 0)).
                    eval_x86_cmp(s, Literal(0), old, width)
                val = unary_builder(a, old)
                if isinstance(args[0], Register):
                    val = s.set_reg(args[0], val)
                    if flags_kind == FLAGS_KEEP_C:
                        set_x86_flags_from_result(s, val, width, set_c_v=False)
                        s.set_reg(cls._flag_v, fn_op("M2C_OVERFLOW", [val], Type.boolean()))
                else:
                    if flags_kind == FLAGS_KEEP_C:
                        set_x86_flags_from_result(s, val, width, set_c_v=False)
                        s.set_reg(cls._flag_v, fn_op("M2C_OVERFLOW", [val], Type.boolean()))
                    write_dst(s, a, val, width_type(width))

            eval_fn = eval_unary
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

            def eval_setcc(s: NodeState, a: InstrArgs) -> None:
                cond = condition_from_expr(a.regs[flag])
                if negated:
                    cond = cond.negated()
                # setcc writes a 0/1 byte; for register destinations this is
                # modeled as writing the full register (usually zeroed
                # beforehand by `xor r, r`).
                val = Cast(expr=cond, reinterpret=False, silent=True, type=Type.u8())
                write_dst(s, a, val, Type.int_of_size(8))

            eval_fn = eval_setcc
        elif base == "cdq":
            assert not args
            inputs = [EAX]
            outputs = [EDX]
            is_effectful = False
            eval_fn = lambda s, a: s.set_reg(
                EDX, BinaryOp.sint(a.regs[EAX], ">>", Literal(31))
            )
        elif base in ("mul", "imul", "div", "idiv") and len(args) <= 1:
            # One-operand forms operate on edx:eax.
            inputs = [EAX] if base in ("mul", "imul") else [EAX, EDX]
            if args:
                src_operand(args[0])
            outputs = [EAX, EDX]
            clobbers = list(cls.flag_regs)
            is_effectful = False

            def eval_muldiv(s: NodeState, a: InstrArgs) -> None:
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

            eval_fn = eval_muldiv
        elif base == "imul":
            # Two/three-operand forms only write the destination register.
            assert len(args) in (2, 3) and isinstance(args[0], Register)
            inputs = [args[0]] if len(args) == 2 else []
            for src in args[1:]:
                src_operand(src)
            outputs = [args[0]]
            clobbers = list(cls.flag_regs)
            is_effectful = False

            def eval_imul(s: NodeState, a: InstrArgs) -> None:
                assert isinstance(args[0], Register)
                if len(args) == 2:
                    lhs: Expression = a.regs[args[0]]
                    rhs = op_value(a, 1, width)
                else:
                    lhs = op_value(a, 1, width)
                    rhs = op_value(a, 2, width)
                s.set_reg(args[0], fold_mul_chains(BinaryOp.int(lhs, "*", rhs)))

            eval_fn = eval_imul
        elif base == "xchg":
            assert len(args) == 2
            for arg in args:
                src_operand(arg)
            for arg in args:
                dest_operand(arg, also_read=True)
            is_effectful = is_store

            def eval_xchg(s: NodeState, a: InstrArgs) -> None:
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
                                store, src_reg if src_reg is not None else EAX
                            )

            eval_fn = eval_xchg
        elif mnemonic in cls.instrs_string:
            str_inputs, str_outputs, is_load, is_store = cls.instrs_string[mnemonic]
            inputs = list(str_inputs)
            outputs = list(str_outputs)
        elif base in cls.instrs_ignore:
            is_effectful = False
            eval_fn = None
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
        if not fn_sig.params_known or fn_sig.is_variadic:
            for i in range(8):
                candidate_slots.append(
                    AbiArgSlot(
                        ArgLoc(offset + 4 * i, len(known_slots) + i, None),
                        Type.any_reg(),
                    )
                )
        return Abi(arg_slots=known_slots, possible_slots=candidate_slots)

    def function_return(self, expr: Expression) -> Dict[Register, Expression]:
        # Return values are in eax, with edx holding the high half of u64's.
        return {
            EAX: as_type(expr, Type.intptr(), silent=True, unify=False),
            EDX: fn_op("SECOND_REG", [expr], Type.reg32(likely_float=False)),
        }
