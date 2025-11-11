from __future__ import annotations

from ..asm_instruction import Register
from ..ir_pattern import IrMatch, IrPattern


class FloatishToSintIrPattern(IrPattern):
    # This pattern handles converting either f32 or f64 into a signed int
    # The `fctiwz` instruction does all the work; this pattern is just to
    # elide the stack store/load pair.
    replacement = "fctiwz.fictive $i, $f"
    parts = [
        "fctiwz $t, $f",
        "stfd $t, (N-4)($r1)",
        "lwz $i, N($r1)",
    ]


class CheckConstantMixin:
    def check(self, m: IrMatch) -> bool:
        # TODO: Also validate that `K($k)` is the expected constant in rodata
        return m.symbolic_registers["k"] in (Register("r2"), Register("r13"))


class SintToDoubleIrPattern(IrPattern, CheckConstantMixin):
    # The replacement asm for these patterns reference the float constant `K($k)`
    # as an input, even though the value is ignored. This is needed to mark `$k`
    # as an input to the pattern for matching.
    replacement = "cvt.d.i.fictive $f, $i, K($k)"
    parts = [
        "lis $a, 0x4330",
        "stw $a, N($r1)",
        "xoris $b, $i, 0x8000",
        "stw $b, (N+4)($r1)",
        "lfd $d, N($r1)",
        "lfd $c, K($k)",
        "fsub $f, $d, $c",
    ]


class UintToDoubleIrPattern(IrPattern, CheckConstantMixin):
    replacement = "cvt.d.u.fictive $f, $i, K($k)"
    parts = [
        "lis $a, 0x4330",
        "stw $a, N($r1)",
        "stw $i, (N+4)($r1)",
        "lfd $d, N($r1)",
        "lfd $c, K($k)",
        "fsub $f, $d, $c",
    ]


class SintToFloatIrPattern(IrPattern, CheckConstantMixin):
    replacement = "cvt.s.i.fictive $f, $i, K($k)"
    parts = [
        "lis $a, 0x4330",
        "stw $a, N($r1)",
        "xoris $b, $i, 0x8000",
        "stw $b, (N+4)($r1)",
        "lfd $d, N($r1)",
        "lfd $c, K($k)",
        "fsubs $f, $d, $c",
    ]


class UintToFloatIrPattern(IrPattern, CheckConstantMixin):
    replacement = "cvt.s.u.fictive $f, $i, K($k)"
    parts = [
        "lis $a, 0x4330",
        "stw $a, N($r1)",
        "stw $i, (N+4)($r1)",
        "lfd $d, N($r1)",
        "lfd $c, K($k)",
        "fsubs $f, $d, $c",
    ]


__all__ = [
    "FloatishToSintIrPattern",
    "CheckConstantMixin",
    "SintToDoubleIrPattern",
    "UintToDoubleIrPattern",
    "SintToFloatIrPattern",
    "UintToFloatIrPattern",
]
