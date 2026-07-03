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
`X86Arch._parse_fpu`. See the design spec for the full rationale.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .asm_file import AsmData
from .asm_instruction import (
    Argument,
    AsmAddressMode,
    AsmLiteral,
    JumpTarget,
    Register,
)
from .asm_pattern import AsmMatcher, AsmPattern, BodyPart, Replacement
from .error import DecompFailure
from .instruction import ArchAsm, Instruction, InstructionMeta


# x87 depth effects, keyed by the base mnemonic (after any width suffix is
# split off). See the spec's §1.2 table.
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
# silently-wrong code (none observed in the target corpus).
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
    to repair (see §4.3). Carries the faulting body index and the failure kind
    so the retry loop can attribute it to a call."""

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
    from .arch_x86 import call_target_symbol, split_width_suffix

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
    from .arch_x86 import split_width_suffix

    for j in range(index - 1, max(index - 8, -1), -1):
        part = body[j]
        if not isinstance(part, Instruction):
            return False
        base, _ = split_width_suffix(part.mnemonic)
        if is_fpu_mnemonic(base):
            return base not in (
                "fst", "fstp", "fistp", "fcom", "fcomp", "fcompp", "fucom",
                "fucomp", "fucompp", "ftst", "ficom", "ficomp", "fnstsw",
                "fstsw", "fldcw", "fstcw", "fnstcw", "fxam",
            )
        if part.function_target is not None or part.jump_target is not None or part.is_return:
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
    keyed = [
        (i, _call_key(body, i)) for i in range(len(body)) if _call_key(body, i)
    ]
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
        # Cheap early-out: the ~3100 functions with no x87 pay only this scan.
        from .arch_x86 import split_width_suffix

        if not any(
            isinstance(part, Instruction)
            and is_fpu_mnemonic(split_width_suffix(part.mnemonic)[0])
            for part in matcher.input
        ):
            return None
        # Seed per-callee deltas from the user context (float/double-returning
        # functions leave their result on the FPU stack), then infer the rest
        # structurally by BFS over delta assignments.
        seed: Dict[str, int] = dict(
            getattr(matcher.arch, "context_fpu_call_deltas", {})
        )
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
    from .arch_x86 import call_target_symbol, split_width_suffix, switch_jump_table_labels

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
        0. Inferred per callee (direct calls) or per site (indirect calls);
        see _call_key / _candidate_moves / the BFS."""
        if item.function_target is None or base != "call":
            return 0
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

    def arg_window_offset(store_index: int, addr: Argument) -> Optional[int]:
        """If a store to `addr` at `store_index` targets an esp-relative slot
        inside the very next call's outgoing-argument window, the frame offset
        of that slot; else None. This is how float arguments reach a call: a
        `push`/`sub esp` allocates the slot and `fstp [esp+k]` fills it (§4.5).
        Only stores with no intervening esp change / barrier qualify, so the
        offset shares the call's frame coordinate."""
        if not (
            isinstance(addr, AsmAddressMode)
            and addr.base is not None
            and addr.base.register_name == "esp"
            and isinstance(addr.addend, AsmLiteral)
        ):
            return None
        offset = addr.addend.value
        for j in range(store_index + 1, len(body)):
            nxt = body[j]
            if not isinstance(nxt, Instruction):
                return None  # a label / merge: give up
            nbase, _ = split_width_suffix(nxt.mnemonic)
            if nbase == "call" and nxt.function_target is not None:
                if len(nxt.args) < 3 or not isinstance(nxt.args[1], AsmLiteral):
                    return None
                win_base = nxt.args[1].value
                assert isinstance(nxt.args[2], AsmLiteral)
                win_bytes = nxt.args[2].value
                in_window = offset >= win_base and (
                    win_bytes < 0 or offset < win_base + win_bytes
                )
                return offset if in_window else None
            # Any other store to a different slot is fine to skip over, but a
            # branch or esp change means this store is not a pending argument.
            if (
                nxt.jump_target is not None
                or nxt.is_return
                or nbase in ("push", "pop", "add", "sub", "call")
            ):
                return None
        return None

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
            delta = call_delta(i, part, base)
            if delta != 0:
                fpret_out = depth if delta == 1 else -1
                fconsume_in = depth - 1 if delta == -1 else -1
                emit(
                    "call",
                    list(part.args)
                    + [AsmLiteral(fpret_out), AsmLiteral(fconsume_in)],
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
        elif base in ("fst", "fstp", "fistp"):
            st_i = _st_index(args[0])
            win_off = (
                arg_window_offset(i, args[0])
                if base in ("fst", "fstp") and st_i is None
                else None
            )
            if st_i is not None:
                # Register store form (`fst st(i)` / `fstp st(i)`).
                if base == "fstp" and st_i == 0:
                    emit("fpop", [flat(0)], meta)  # discard-pop idiom
                elif base == "fstp":
                    emit("fmovpop", [flat(st_i), flat(0)], meta)
                else:
                    emit("fmov", [flat(st_i), flat(0)], meta)
            elif win_off is not None:
                # A float stored into the next call's argument window (§4.5).
                _, w = split_width_suffix(mnemonic)
                stem = "fstparg" if base == "fstp" else "fstarg"
                mn = stem if w == 4 else stem + ".q"
                emit(mn, [AsmLiteral(win_off), flat(0)], meta)
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

        # --- Compares (slice 2), control word (slice 3), transcendentals
        # (slice 5): pass fictive top-of-stack forms to the eval layer, which
        # raises a clean DecompFailure for anything not yet implemented. ---
        elif base in ("fcom", "fcomp", "fucom", "fucomp"):
            src: Argument
            if args and _st_index(args[0]) is None:
                src = args[0]  # memory operand
            elif args:
                src = flat(_st_index(args[0]) or 0)
            else:
                src = flat(1)  # bare form compares st0 with st1
            emit(base, [flat(0), src], meta)
        elif base in ("fcompp", "fucompp"):
            emit(base, [flat(0), flat(1)], meta)
        elif base == "ftst":
            emit(base, [flat(0)], meta)
        elif base in ("ficom", "ficomp"):
            emit(base, [flat(0), args[0]], meta)
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
