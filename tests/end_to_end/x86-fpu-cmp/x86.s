# x87 compares: real MSVC6 /O2 output of
#     int in_range(float x, float lo, float hi) { return x > lo && x < hi; }
# The fcomp/fnstsw/test-ah/jcc idiom, exercising both the 0x41 mask (C0|C3,
# "st0 > src") and the 0x01 mask (C0, "st0 >= src") plus && short-circuiting.
# (Float negations are not simplified past NaN, so x < hi surfaces as the
# honest !(x >= hi).)
test:
    FLD dword ptr [ESP + 0x4]
    FCOMP dword ptr [ESP + 0x8]
    FNSTSW AX
    TEST AH, 0x41
    JNZ _LAB_zero
    FLD dword ptr [ESP + 0x4]
    FCOMP dword ptr [ESP + 0xc]
    FNSTSW AX
    TEST AH, 0x1
    JZ _LAB_zero
    MOV EAX, 0x1
    RET
_LAB_zero:
    XOR EAX, EAX
    RET
