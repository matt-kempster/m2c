# x87 float call ABI: real MSVC6 /O2 output of
#     extern float mix(float u, float v);
#     float apply(float a, float b) { return mix(a * b, a - b) + a; }
# Each float argument is allocated with `push ecx` and filled with `fstp
# [esp]` into the call's argument window (routed to a subroutine arg);
# the callee returns its float in st(0) (call depth delta +1, inferred), which
# is then added to a and returned.
test:
    FLD dword ptr [ESP + 0x4]
    FSUB dword ptr [ESP + 0x8]
    PUSH ECX
    FSTP dword ptr [ESP]
    FLD dword ptr [ESP + 0x8]
    FMUL dword ptr [ESP + 0xc]
    PUSH ECX
    FSTP dword ptr [ESP]
    CALL _mix
    FADD dword ptr [ESP + 0xc]
    ADD ESP, 0x8
    RET
