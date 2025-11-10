from __future__ import annotations

from typing import Optional

from ..asm_instruction import (
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmInstruction,
    AsmLiteral,
    Register,
)
from ..asm_pattern import (
    AsmMatch,
    AsmMatcher,
    AsmPattern,
    Replacement,
    SimpleAsmPattern,
    make_pattern,
)
from ..instruction import Instruction


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


class MfcrPattern(SimpleAsmPattern):
    """Comparison results extracted as ints."""

    pattern = make_pattern(
        "mfcr $x",
        "rlwinm $x, $x, N, 31, 31",
    )

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        x = m.regs["x"]
        if m.literals["N"] == 1:
            reg = Register("cr0_lt")
        elif m.literals["N"] == 2:
            reg = Register("cr0_gt")
        elif m.literals["N"] == 3:
            reg = Register("cr0_eq")
        elif m.literals["N"] == 4:
            reg = Register("cr0_so")
        else:
            return None
        return Replacement([AsmInstruction("move.fictive", [x, reg])], len(m.body))


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
            and not matcher.is_local_label(instr.args[0].symbol_name)
        ):
            return Replacement(
                [
                    AsmInstruction("bl", instr.args),
                    AsmInstruction("blr", []),
                ],
                1,
            )
        return None


class SaveRestoreRegsFnPattern(AsmPattern):
    """Expand calls to MWCC's built-in `_{save,rest}{gpr,fpr}_` functions into
    register saves/restores."""

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        bl = matcher.input[matcher.index]
        if (
            not isinstance(bl, Instruction)
            or bl.mnemonic != "bl"
            or not isinstance(bl.args[0], AsmGlobalSymbol)
        ):
            return None
        parts = bl.args[0].symbol_name.split("_")
        if len(parts) != 3 or parts[0]:
            return None
        if parts[1] in ("savegpr", "restgpr"):
            mnemonic = "stw" if parts[1] == "savegpr" else "lwz"
            size = 4
            reg_prefix = "r"
        elif parts[1] in ("savefpr", "restfpr"):
            mnemonic = "stfd" if parts[1] == "savefpr" else "lfd"
            size = 8
            reg_prefix = "f"
        else:
            return None

        # Find "addi $r11, $r1, N" above, with perhaps some instructions in between.
        for i in range(matcher.index - 1, -1, -1):
            instr = matcher.input[i]
            if (
                isinstance(instr, Instruction)
                and instr.mnemonic == "addi"
                and instr.args[0] == Register("r11")
                and instr.args[1] == Register("r1")
                and isinstance(instr.args[2], AsmLiteral)
            ):
                addend = instr.args[2].value
                break
        else:
            return None

        regnum = int(parts[2])
        new_instrs = []
        for i in range(regnum, 32):
            reg = Register(reg_prefix + str(i))
            stack_pos = AsmAddressMode(
                base=Register("r1"),
                addend=AsmLiteral(size * (i - 32) + addend),
                writeback=None,
            )
            new_instrs.append(AsmInstruction(mnemonic, [reg, stack_pos]))
        return Replacement(new_instrs, 1)


class BoolCastPattern(SimpleAsmPattern):
    """Cast to bool (a 1 bit type in MWCC), which also can be emitted from `!!x`."""

    pattern = make_pattern(
        "neg $a, $x",
        "addic $r0, $a, -1",
        "subfe $r0, $r0, $a",
    )

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        boolcast = AsmInstruction("boolcast.fictive", [Register("r0"), m.regs["x"]])
        if m.regs["a"] == Register("r0"):
            return None
        elif m.regs["x"] == m.regs["a"]:
            return Replacement([boolcast, m.body[0]], len(m.body))
        else:
            return Replacement([m.body[0], boolcast], len(m.body))


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


class FloatishToUintPattern(SimpleAsmPattern):
    pattern = make_pattern("bl __cvt_fp2unsigned")

    def replace(self, m: AsmMatch) -> Optional[Replacement]:
        return Replacement(
            [AsmInstruction("cvt.u.d.fictive", [Register("r3"), Register("f1")])],
            len(m.body),
        )


class StructCopyPattern(AsmPattern):
    """Recognizing struct copy when it starts with lwz lwz stw stw. Others
    would cause false positves. Maybe we can find another way for those using
    context?
    This pattern appears on almost every GC and Wii MW compiler version when using C
    and GC MW 1.0-1.2.5n when using C++.
    """

    pattern = make_pattern(
        "lwz $a, I($s)",
        "lwz $b, (I+4)($s)",
        "stw $a, I($d)",
        "stw $b, (I+4)($d)",
    )

    def match(self, matcher: AsmMatcher) -> Optional[Replacement]:
        # Use the initial patterns first
        m = matcher.try_match(self.pattern)
        if m is None:
            return None
        i = 8
        pattern_ext = self.pattern.copy()
        while True:
            pattern2 = make_pattern(
                f"lwz $a, (I+{i})($s)",
                f"lwz $b, (I+{i+4})($s)",
                f"stw $a, (I+{i})($d)",
                f"stw $b, (I+{i+4})($d)",
            )

            m2 = matcher.try_match(pattern_ext + pattern2)
            if m2:
                m = m2
                i += 8
                pattern_ext.extend(pattern2)
            else:
                # Unaligned struct
                pattern_end_4b = make_pattern(
                    f"lwz $b, (I+{i})($s)",
                    f"stw $b, (I+{i})($d)",
                )
                m_end = matcher.try_match(pattern_ext + pattern_end_4b)
                if m_end:
                    m = m_end
                    i += 4
                    pattern_ext.extend(pattern_end_4b)

                pattern_end_2b = make_pattern(
                    f"lhz $b, (I+{i})($s)",
                    f"sth $b, (I+{i})($d)",
                )
                m_end = matcher.try_match(pattern_ext + pattern_end_2b)
                if m_end:
                    m = m_end
                    i += 2
                    pattern_ext.extend(pattern_end_2b)

                pattern_end_1b = make_pattern(
                    f"lbz $b, (I+{i})($s)",
                    f"stb $b, (I+{i})($d)",
                )
                m_end = matcher.try_match(pattern_ext + pattern_end_1b)
                if m_end:
                    m = m_end
                    i += 1
                break

        return Replacement(
            [
                AsmInstruction(
                    "structcopy.fictive", [m.regs["d"], m.regs["s"], AsmLiteral(i)]
                )
            ],
            len(m.body),
        )


__all__ = [
    "FcmpoCrorPattern",
    "MfcrPattern",
    "TailCallPattern",
    "SaveRestoreRegsFnPattern",
    "BoolCastPattern",
    "BranchCtrPattern",
    "FloatishToUintPattern",
    "StructCopyPattern",
]
