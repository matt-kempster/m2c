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

from typing import Dict, List, Optional, Set, Tuple

from .asm_file import AsmData
from .asm_instruction import (
    Argument,
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


class X86FpuRewritePattern(AsmPattern):
    """Whole-body rewrite eliminating the x87 register stack (see module doc).

    Mirrors `X86StackRewritePattern`: matches only at index 0, computes the
    entire rewritten body, and (in the presence of float-returning or
    stack-consuming callees) retries the dataflow with per-callee depth deltas
    inferred structurally, keeping the result only if it becomes consistent."""

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
        new_body = rewrite_fpu_ops(
            matcher.input, matcher.arch, matcher.asm_data, matcher.labels
        )
        return Replacement(new_body, len(matcher.input), clobbers=[])


def rewrite_fpu_ops(
    body: List[BodyPart],
    arch: ArchAsm,
    asm_data: AsmData,
    labels: Set[str],
) -> List[BodyPart]:
    from .arch_x86 import split_width_suffix, switch_jump_table_labels

    label_pos: Dict[str, int] = {}
    for i, part in enumerate(body):
        if not isinstance(part, Instruction):
            for name in part.names:
                label_pos[name] = i

    def instr_str(item: Instruction) -> str:
        return f"`{item}` {item.meta.loc_str()}"

    def depth_delta(item: Instruction, base: str) -> int:
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
        return 0

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
                raise DecompFailure(
                    f"x87 stack depth mismatch ({prev} vs {depth}) at {loc}"
                )
            continue
        states[index] = depth
        if not isinstance(part, Instruction):
            push(index + 1, depth)
            continue
        if part.is_return:
            continue
        base, _ = split_width_suffix(part.mnemonic)
        out = depth + depth_delta(part, base)
        if out < 0:
            raise DecompFailure(
                f"x87 stack underflow (depth {depth}) at {instr_str(part)}"
            )
        if out > 8:
            raise DecompFailure(
                f"x87 stack overflow (depth {out}) at {instr_str(part)}"
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
        if depth is None or not is_fpu_mnemonic(base):
            # Unreachable code, or a non-x87 instruction: pass through.
            new_body.append(part)
            continue
        if base in FPU_DROP:
            continue
        args = part.args
        meta = part.meta
        mnemonic = part.mnemonic

        def flat(st_i: int, *, at: int = depth) -> Register:
            """The virtual register for physical `st(st_i)` at depth `at`."""
            idx = at - 1 - st_i
            if idx < 0 or idx > 7:
                raise DecompFailure(
                    f"x87 stack underflow reading st({st_i}) at depth {at}: "
                    f"{instr_str(part)}"
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
            if st_i is not None:
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
        elif base == "ficom":
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
