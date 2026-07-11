"""x87 FPU register-stack elimination for the x86 backend.

The x87 FPU is a register *stack*: instructions name registers relative to the
current top (`st(0)`..`st(7)`), and push/pop operations shift what each name
refers to. m2c's translation layer assumes fixed registers, so -- exactly as
`X86StackRewritePattern` does for the moving `esp` -- a whole-body prepass
computes the x87 stack depth at every instruction by linear dataflow and
rewrites the stack-relative `st(i)` names into fixed virtual registers before
translation. The rest of m2c then never learns that x87 exists.

Virtual registers are bottom-anchored: at depth `d`, physical `st(i)` denotes
the value pushed `(d-1-i)` pushes ago and is rewritten to the flat register
`f{d-1-i}`. So the bottom slot is always `f0`, a value keeps one stable name
for its whole lifetime regardless of pushes/pops above it, and `fld` at depth
`d` defines `f{d}`. Popping instructions *kill* their virtual register (it
stops existing), which is what makes float-return detection sound: `f0` is set
at a return block iff the stack is non-empty there.

This pass runs after `X86StackRewritePattern`, so every x87 memory operand is
already frame-resolved and every `call` already carries its argument-window
annotation. It emits fictive instructions carrying explicit virtual-register
arguments (e.g. `fadd $f1, $f0`, `fstp.s [m], $f2`); their semantics live in
`X86Arch._parse_fpu`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple, cast

from .asm_file import AsmData
from .asm_instruction import (
    Argument,
    AsmLiteral,
    JumpTarget,
    Register,
)
from .asm_pattern import AsmMatcher, AsmPattern, BodyPart, Replacement
from .error import DecompFailure
from .instruction import ArchAsm, Instruction, InstructionMeta
from .x86_utils import (
    call_target_symbol,
    split_width_suffix,
    switch_jump_table_labels,
)

if TYPE_CHECKING:
    from .arch_x86 import X86Arch


# x87 depth effects, keyed by the base mnemonic (after any width suffix is
# split off).
FPU_PUSH: Set[str] = {
    "fld",
    "fild",
    "fld1",
    "fldz",
    "fldpi",
    "fldl2e",
    "fldl2t",
    "fldlg2",
    "fldln2",
}
FPU_POP1: Set[str] = {
    "fstp",
    "fstparg",
    "fistp",
    "fcomp",
    "fucomp",
    "ficomp",
    "faddp",
    "fsubp",
    "fsubrp",
    "fmulp",
    "fdivp",
    "fdivrp",
    "fpatan",
    "fyl2x",
    "fyl2xp1",
}
FPU_POP2: Set[str] = {"fcompp", "fucompp"}
FPU_NEUTRAL: Set[str] = {
    "fadd",
    "fsub",
    "fsubr",
    "fmul",
    "fdiv",
    "fdivr",
    "fiadd",
    "fisub",
    "fisubr",
    "fimul",
    "fidiv",
    "fidivr",
    "fcom",
    "fucom",
    "ficom",
    "ftst",
    "fst",
    "fstarg",
    "fist",
    "fchs",
    "fabs",
    "fsqrt",
    "fsin",
    "fcos",
    "frndint",
    "fscale",
    "f2xm1",
    "fprem",
    "fprem1",
    "fxch",
    "fnstsw",
    "fstsw",
    "fldcw",
    "fstcw",
    "fnstcw",
    "fxam",
}
# Depth-neutral no-ops we drop entirely (they carry no value semantics).
FPU_DROP: Set[str] = {"fwait", "wait", "fnop"}
# Instructions we deliberately do not support; fail loudly rather than emit
# silently-wrong code (not observed in typical MSVC6 output).
FPU_UNSUPPORTED: Set[str] = {
    "fptan",
    "fsincos",
    "fbld",
    "fbstp",
    "fsave",
    "fnsave",
    "frstor",
    "fstenv",
    "fnstenv",
    "fldenv",
    "fincstp",
    "fdecstp",
    "ffree",
    "ffreep",
    "finit",
    "fninit",
    "fclex",
    "fnclex",
    "fisttp",
}

ALL_FPU: Set[str] = (
    FPU_PUSH | FPU_POP1 | FPU_POP2 | FPU_NEUTRAL | FPU_DROP | FPU_UNSUPPORTED
)

# CRT helpers that consume st(0) as their single argument (x87 stack delta -1)
# and return an integer in eax/edx, so they must be seeded unconditionally: an
# `fld; call __ftol` pair stays depth-balanced with delta 0, so no dataflow
# fault would ever trigger the structural ftol-shaped repair, and the pushed
# value would be silently dropped. MSVC6 emits `__ftol` for float/double->long
# (and __int64) casts when /QIfist is off; `__ftol2` is the later CRT spelling.
X86_FPU_HELPER_DELTAS: Dict[str, int] = {
    "__ftol": -1,
    "__ftol2": -1,
}

# MSVC6 /Oi lowers the libm functions that have no single x87 instruction into
# calls to CRT helpers that take their arguments on, and return their result
# on, the x87 stack (`fld a; fld b; call __CIpow`). Each maps to a fictive FPU
# op (see FPU_UNARY_OPS / FPU_BINARY_OPS in arch_x86) carrying the real call, so
# the stack effect is modeled exactly as (consumed, pushed) -- not the ±1
# single-value delta, which would drop one operand of a two-argument helper and
# silently miscompile `pow(a, b)` to one of its arguments. The value is
# (fictive mnemonic, argument count); two-argument helpers are net depth -1,
# one-argument helpers net 0.
X86_FPU_CALL_HELPERS: Dict[str, Tuple[str, int]] = {
    "__CIpow": ("ci_pow.fictive", 2),
    "__CIfmod": ("ci_fmod.fictive", 2),
    "__CIatan2": ("ci_atan2.fictive", 2),
    "__CIsqrt": ("ci_sqrt.fictive", 1),
    "__CIsin": ("ci_sin.fictive", 1),
    "__CIcos": ("ci_cos.fictive", 1),
    "__CItan": ("ci_tan.fictive", 1),
    "__CIexp": ("ci_exp.fictive", 1),
    "__CIlog": ("ci_log.fictive", 1),
    "__CIlog10": ("ci_log10.fictive", 1),
    "__CIasin": ("ci_asin.fictive", 1),
    "__CIacos": ("ci_acos.fictive", 1),
    "__CIatan": ("ci_atan.fictive", 1),
    "__CIsinh": ("ci_sinh.fictive", 1),
    "__CIcosh": ("ci_cosh.fictive", 1),
    "__CItanh": ("ci_tanh.fictive", 1),
}


def is_fpu_mnemonic(base: str) -> bool:
    return base in ALL_FPU


def _st_index(arg: Argument) -> Optional[int]:
    """The N of an `st(N)` register operand (spelled `stN` after parsing), or
    None if `arg` is not an x87 stack register."""
    if isinstance(arg, Register):
        name = arg.register_name
        if name.startswith("st") and name[2:].isdigit():
            return int(name[2:])
    return None


class X87StackError(DecompFailure):
    """A dataflow inconsistency that per-callee depth-delta inference can try
    to repair (see X86FpuRewritePattern.match). Carries the faulting body
    index and the failure kind so the retry loop can attribute it to a call."""

    def __init__(self, message: str, index: int, kind: str) -> None:
        super().__init__(message)
        self.index = index
        self.kind = kind  # "underflow" | "overflow" | "conflict"


# Bound on the number of candidate delta-assignments explored per function.
MAX_FPU_INFER_STATES = 200
# How many of the calls nearest a fault to branch on when exploring.
FPU_INFER_FANOUT = 5


def _call_key(body: List[BodyPart], index: int) -> Optional[str]:
    """A stable key for the callee's depth delta: its symbol for a direct call
    (so all sites of a callee agree), or a per-site `@index` for an indirect
    call through a register/pointer (a float-returning function-pointer
    argument, which has no symbol to share). None if not a call."""
    part = body[index]
    if not (
        isinstance(part, Instruction)
        and part.function_target is not None
        and split_width_suffix(part.mnemonic)[0] == "call"
    ):
        return None
    sym = call_target_symbol(part.args[0])
    return sym if sym is not None else f"@{index}"


def _is_ftol_shaped(body: List[BodyPart], index: int) -> bool:
    """Whether the call at `index` looks like an ftol-style helper that
    consumes st(0): a value-producing FPU op (arith/load, not a store or
    compare) sits just before it with no barrier between. Such a call leaves
    the FPU stack imbalanced unless its delta is -1, but the resulting conflict
    can surface far away, so these are always offered to the search."""
    for j in range(index - 1, max(index - 8, -1), -1):
        part = body[j]
        if not isinstance(part, Instruction):
            return False
        base, _ = split_width_suffix(part.mnemonic)
        if is_fpu_mnemonic(base):
            return base not in (
                "fst",
                "fstp",
                "fstarg",
                "fstparg",
                "fistp",
                "fcom",
                "fcomp",
                "fcompp",
                "fucom",
                "fucomp",
                "fucompp",
                "ftst",
                "ficom",
                "ficomp",
                "fnstsw",
                "fstsw",
                "fldcw",
                "fstcw",
                "fnstcw",
                "fxam",
            )
        if (
            part.function_target is not None
            or part.jump_target is not None
            or part.is_return
        ):
            return False
    return False


def _candidate_moves(
    body: List[BodyPart], err: X87StackError, call_deltas: Dict[str, int]
) -> List[Tuple[str, int]]:
    """Delta adjustments worth exploring to repair a dataflow failure: the
    calls nearest the fault, each toward the failure-appropriate sign first
    (underflow -> a float-returning callee, +1; overflow -> a stack-consuming
    ftol-style helper, -1; a conflict is ambiguous, so try both), plus any
    ftol-shaped call anywhere in the body toward -1. Feeds a BFS, so a wrong
    guess is explored in parallel with the right one rather than committed."""
    keyed = [(i, _call_key(body, i)) for i in range(len(body)) if _call_key(body, i)]
    keyed.sort(key=lambda p: (abs(p[0] - err.index), p[0]))
    signs = [-1, 1] if err.kind == "overflow" else [1, -1]
    moves: List[Tuple[str, int]] = []

    def offer(key: str, delta: int) -> None:
        if call_deltas.get(key, 0) != delta and (key, delta) not in moves:
            moves.append((key, delta))

    for delta in signs:
        for _, key in keyed[:FPU_INFER_FANOUT]:
            assert key is not None
            offer(key, delta)
    for i, key in keyed:
        if key is not None and _is_ftol_shaped(body, i):
            offer(key, -1)
    return moves


class X86FpuRewritePattern(AsmPattern):
    """Whole-body rewrite eliminating the x87 register stack (see module doc).

    Mirrors `X86StackRewritePattern`: matches only at index 0 and computes the
    entire rewritten body. In the presence of float-returning or
    stack-consuming callees, it searches (breadth-first, minimal changes first)
    for a per-callee depth-delta assignment that makes the dataflow consistent,
    keyed per-symbol so all call sites of a callee agree."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        if matcher.index != 0:
            return None
        # Seed per-callee deltas from the known stack-consuming CRT helpers and
        # the user context (float/double-returning functions leave their result
        # on the FPU stack), then infer the rest structurally by BFS over delta
        # assignments. Context deltas win on conflict.
        # This pattern is only registered on X86Arch, whose load_context
        # mines the deltas from context prototypes.
        arch = cast("X86Arch", matcher.arch)
        seed: Dict[str, int] = dict(X86_FPU_HELPER_DELTAS)
        seed.update(arch.context_fpu_call_deltas)
        # Cheap early-out: functions with no x87 instructions pay only this scan.
        # A function still needs the pass, though, if it calls a context-known
        # float/double-returning function -- even with no x87 instruction of
        # its own (a forwarding wrapper like `call _returns_float; ret`), the
        # call must be annotated as producing st(0).
        if not any(
            isinstance(part, Instruction)
            and is_fpu_mnemonic(split_width_suffix(part.mnemonic)[0])
            for part in matcher.input
        ) and not (
            seed
            and any(
                _call_key(matcher.input, i) in seed for i in range(len(matcher.input))
            )
        ):
            return None
        queue: List[Dict[str, int]] = [seed]
        visited: Set[FrozenSet[Tuple[str, int]]] = {frozenset(seed.items())}
        last_err: Optional[X87StackError] = None
        states_tried = 0
        while queue and states_tried < MAX_FPU_INFER_STATES:
            call_deltas = queue.pop(0)
            states_tried += 1
            try:
                new_body = rewrite_fpu_ops(
                    matcher.input,
                    matcher.arch,
                    matcher.asm_data,
                    matcher.labels,
                    call_deltas,
                )
                return Replacement(new_body, len(matcher.input), clobbers=[])
            except X87StackError as e:
                last_err = e
                for sym, delta in _candidate_moves(matcher.input, e, call_deltas):
                    nxt = dict(call_deltas)
                    nxt[sym] = delta
                    key = frozenset(nxt.items())
                    if key not in visited:
                        visited.add(key)
                        queue.append(nxt)
        assert last_err is not None
        raise last_err


def rewrite_fpu_ops(
    body: List[BodyPart],
    arch: ArchAsm,
    asm_data: AsmData,
    labels: Set[str],
    call_deltas: Optional[Dict[str, int]] = None,
) -> List[BodyPart]:
    call_deltas = call_deltas or {}

    label_pos: Dict[str, int] = {}
    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            for name in part.names:
                label_pos[name] = i

    def instr_str(item: Instruction) -> str:
        return f"`{item}` {item.meta.loc_str()}"

    def call_delta(index: int, item: Instruction, base: str) -> int:
        """The FPU stack effect of a call: +1 if the callee returns a float in
        st(0), -1 if it consumes st(0) (a hand-written ftol-style helper), else
        0. A CRT `__CIxxx` math helper has a fixed (consumed, pushed) effect (a
        two-argument helper is net -1, one-argument net 0); an unrecognized
        `__CI*` helper fails loud rather than being guessed by the ±1 inference,
        which could accept a depth-consistent but value-wrong assignment. Other
        callees are inferred per callee (direct) or per site (indirect); see
        _call_key / _candidate_moves / the BFS."""
        if item.function_target is None or base != "call":
            return 0
        sym = call_target_symbol(item.args[0]) if item.args else None
        if sym is not None and sym.startswith("__CI"):
            helper = X86_FPU_CALL_HELPERS.get(sym)
            if helper is None:
                raise DecompFailure(
                    f"unsupported x87 CRT math helper {sym}: {instr_str(item)}"
                )
            return 1 - helper[1]  # 2-arg -> -1, 1-arg -> 0
        key = _call_key(body, index)
        return call_deltas.get(key, 0) if key is not None else 0

    def depth_delta(index: int, item: Instruction, base: str) -> int:
        if base in FPU_UNSUPPORTED:
            raise DecompFailure(
                f"unsupported x87 instruction {base}: {instr_str(item)}"
            )
        if base in FPU_PUSH:
            return 1
        if base in FPU_POP1:
            return -1
        if base in FPU_POP2:
            return -2
        return call_delta(index, item, base)

    # Pass 1: dataflow computing the x87 stack depth at entry to every
    # reachable instruction. State is a single small int; merges require
    # equality (a disagreement is unbalanced code we refuse loudly).
    states: List[Optional[int]] = [None] * len(body)
    worklist: List[Tuple[int, int]] = [(0, 0)]

    def push(index: int, depth: int) -> None:
        worklist.append((index, depth))

    while worklist:
        index, depth = worklist.pop()
        if index >= len(body):
            continue
        part = body[index]
        prev = states[index]
        if prev is not None:
            if prev != depth:
                loc = (
                    instr_str(part)
                    if isinstance(part, Instruction)
                    else f"label {part}"
                )
                raise X87StackError(
                    f"x87 stack depth mismatch ({prev} vs {depth}) at {loc}",
                    index,
                    "conflict",
                )
            continue
        states[index] = depth
        if not isinstance(part, Instruction):
            push(index + 1, depth)
            continue
        if part.is_return:
            # The x86 ABI leaves the x87 stack empty at a return except for a
            # single float/double result in st(0), so a valid return depth is 0
            # (void/int) or 1 (float/double). A deeper stack means leftover
            # values -- typically a per-callee depth delta that is value-wrong
            # but sign-consistent (e.g. an unseeded `__CI*` helper that consumes
            # two operands and pushes one). Raise a conflict so the BFS repair
            # loop can search call-delta assignments; a genuinely unbalanced
            # body then fails loud instead of silently miscompiling.
            if depth > 1:
                raise X87StackError(
                    f"x87 stack not empty at return (depth {depth}): at most one "
                    f"float/double result may remain in st(0) at {instr_str(part)}",
                    index,
                    "conflict",
                )
            continue
        base, _ = split_width_suffix(part.mnemonic)
        out = depth + depth_delta(index, part, base)
        if out < 0:
            raise X87StackError(
                f"x87 stack underflow (depth {depth}) at {instr_str(part)}",
                index,
                "underflow",
            )
        if out > 8:
            raise X87StackError(
                f"x87 stack overflow (depth {out}) at {instr_str(part)}",
                index,
                "overflow",
            )
        jt = part.jump_target
        if isinstance(jt, JumpTarget):
            if jt.target in label_pos:
                push(label_pos[jt.target], out)
            if part.is_conditional:
                push(index + 1, out)
            continue
        if base == "jmp":
            # Indirect jump: a jump table.
            targets = switch_jump_table_labels(part, asm_data)
            if targets is None:
                raise DecompFailure(
                    f"Unable to determine jump table for {instr_str(part)}"
                )
            for target in targets:
                if target in label_pos:
                    push(label_pos[target], out)
            continue
        push(index + 1, out)

    # Pass 2: emit the rewritten body.
    new_body: List[BodyPart] = []

    def emit(mnemonic: str, emit_args: List[Argument], meta: InstructionMeta) -> None:
        new_body.append(arch.parse(mnemonic, emit_args, meta.derived()))

    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            new_body.append(part)
            continue
        depth = states[i]
        base, _ = split_width_suffix(part.mnemonic)
        # Annotate a call that pushes (delta +1) or consumes (delta -1) a
        # float: append the pushed virtual register index (fpret) and the
        # consumed one (or -1). parse() turns these into call outputs/inputs.
        if (
            depth is not None
            and base == "call"
            and part.function_target is not None
            and len(part.args) == 3
        ):
            sym = call_target_symbol(part.args[0]) if part.args else None
            helper = X86_FPU_CALL_HELPERS.get(sym) if sym is not None else None
            if helper is not None:
                # A CRT math helper (`fld a; [fld b;] call __CIxxx`): rewrite it
                # into its fictive FPU op reading st(0)[/st(1)] and producing the
                # result in place (1-arg) or in st(1) with a pop (2-arg).
                fictive, nargs = helper
                if depth < nargs:
                    raise X87StackError(
                        f"x87 stack underflow (depth {depth}) calling {sym}: "
                        f"{instr_str(part)}",
                        i,
                        "underflow",
                    )
                slots = [Register(f"f{depth - 1 - k}") for k in range(nargs)]
                emit(fictive, list(slots), part.meta)
                continue
            delta = call_delta(i, part, base)
            if delta != 0:
                fpret_out = depth if delta == 1 else -1
                fconsume_in = depth - 1 if delta == -1 else -1
                emit(
                    "call",
                    list(part.args) + [AsmLiteral(fpret_out), AsmLiteral(fconsume_in)],
                    part.meta,
                )
                continue
        if depth is None or not is_fpu_mnemonic(base):
            # Unreachable code, or a non-x87 instruction: pass through.
            new_body.append(part)
            continue
        if base in FPU_DROP:
            continue
        args = part.args
        meta = part.meta
        mnemonic = part.mnemonic

        def flat(st_i: int, *, at: int = depth, fault: int = i) -> Register:
            """The virtual register for physical `st(st_i)` at depth `at`."""
            idx = at - 1 - st_i
            if idx < 0 or idx > 7:
                raise X87StackError(
                    f"x87 stack underflow reading st({st_i}) at depth {at}: "
                    f"{instr_str(part)}",
                    fault,
                    "underflow",
                )
            return Register(f"f{idx}")

        # `flat(0)` (the current top of stack) is only valid at depth >= 1;
        # the pushes below define f{depth} and never read it.

        # --- Pushes: a new value appears at f{depth}. ---
        if base == "fld":
            # `fld m` (load) or `fld st(i)` (duplicate).
            st_i = _st_index(args[0])
            if st_i is not None:
                emit("fmov", [Register(f"f{depth}"), flat(st_i)], meta)
            else:
                emit(mnemonic, [Register(f"f{depth}"), args[0]], meta)
        elif base == "fild":
            emit(mnemonic, [Register(f"f{depth}"), args[0]], meta)
        elif base in ("fld1", "fldz", "fldpi", "fldl2e", "fldl2t", "fldlg2", "fldln2"):
            emit(base, [Register(f"f{depth}")], meta)

        # --- Stores from the top of stack. ---
        elif base in ("fst", "fstp", "fistp", "fstarg", "fstparg"):
            st_i = _st_index(args[0])
            if base in ("fstarg", "fstparg"):
                emit(mnemonic, [args[0], flat(0)], meta)
            elif st_i is not None:
                # Register store form (`fst st(i)` / `fstp st(i)`).
                if base == "fstp" and st_i == 0:
                    emit("fpop", [flat(0)], meta)  # discard-pop idiom
                elif base == "fstp":
                    emit("fmovpop", [flat(st_i), flat(0)], meta)
                else:
                    emit("fmov", [flat(st_i), flat(0)], meta)
            else:
                emit(mnemonic, [args[0], flat(0)], meta)

        # --- fxch: swap two slots (bare form swaps st0 and st1). ---
        elif base == "fxch":
            other = _st_index(args[0]) if args else None
            emit("fxch", [flat(0), flat(other if other is not None else 1)], meta)

        # --- Non-popping arithmetic: dst is st0 for the memory/`st,st(i)`
        # forms, or an explicit st(i) for the `st(i),st` form. ---
        elif base in ("fadd", "fsub", "fsubr", "fmul", "fdiv", "fdivr"):
            if len(args) >= 2:
                dst_i = _st_index(args[0])
                src_i = _st_index(args[1])
                assert dst_i is not None and src_i is not None
                emit(mnemonic, [flat(dst_i), flat(src_i)], meta)
            else:
                emit(mnemonic, [flat(0), args[0]], meta)  # `fadd m`
        elif base in ("fiadd", "fisub", "fisubr", "fimul", "fidiv", "fidivr"):
            emit(mnemonic, [flat(0), args[0]], meta)  # `fiadd m`

        # --- Popping arithmetic: `faddp st(i), st` (bare form => st(1), st). ---
        elif base in ("faddp", "fsubp", "fsubrp", "fmulp", "fdivp", "fdivrp"):
            dst_i = _st_index(args[0]) if args else None
            emit(base, [flat(dst_i if dst_i is not None else 1), flat(0)], meta)

        # --- Unary operations on the top of stack. ---
        elif base in ("fchs", "fabs", "fsqrt", "fsin", "fcos", "frndint", "f2xm1"):
            emit(base, [flat(0)], meta)

        # --- Compares, control word, transcendentals: pass fictive
        # top-of-stack forms to the eval layer, which raises a clean
        # DecompFailure for anything not implemented. ---
        elif base in ("fcom", "fcomp", "fucom", "fucomp"):
            src: Argument
            if args and _st_index(args[0]) is None:
                src = args[0]  # memory operand
            elif args:
                src = flat(_st_index(args[0]) or 0)
            else:
                src = flat(1)  # bare form compares st0 with st1
            # Keep the width-suffixed mnemonic (like fld/fadd): a memory-operand
            # compare (`fcomp qword`) must read the operand at its real width,
            # or an f64 constant/local is truncated to f32.
            emit(mnemonic, [flat(0), src], meta)
        elif base in ("fcompp", "fucompp"):
            emit(base, [flat(0), flat(1)], meta)
        elif base == "ftst":
            emit(base, [flat(0)], meta)
        elif base in ("ficom", "ficomp"):
            emit(mnemonic, [flat(0), args[0]], meta)
        elif base in ("fnstsw", "fstsw"):
            emit("fnstsw", list(args), meta)
        elif base in ("fldcw", "fstcw", "fnstcw"):
            emit(base, list(args), meta)
        elif base in ("fscale", "fprem", "fprem1", "fpatan", "fyl2x", "fyl2xp1"):
            emit(base, [flat(0), flat(1)], meta)
        elif base == "fxam":
            emit(base, [flat(0)], meta)
        else:
            raise DecompFailure(f"unhandled x87 instruction: {base}")

    return new_body
