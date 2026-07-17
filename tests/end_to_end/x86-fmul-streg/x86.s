# One-operand register forms of the non-popping x87 arithmetic ops:
# `fmul st(i)` means `st0 *= st(i)` (likewise fadd/fsub/... st(i)). IDA emits
# these throughout matrix/vector math; they must not be mistaken for the
# memory-operand form (which would read an unset stack slot).
test:
    FLD dword ptr [ESP + 0x4]
    FLD dword ptr [ESP + 0x8]
    FMUL ST(1)
    FADDP
    RET
