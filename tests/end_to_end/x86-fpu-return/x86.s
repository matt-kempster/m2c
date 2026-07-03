# x87 float return: real MSVC6 /O2 output of
#     float lerp(float a, float b, float t) { return a + (b - a) * t; }
# The result is left in st(0) at `ret`; the FPU prepass keeps the bottom slot
# f0 live there, and base_return_regs picks it up (like MIPS v0/f0).
test:
    FLD dword ptr [ESP + 0x8]
    FSUB dword ptr [ESP + 0x4]
    FMUL dword ptr [ESP + 0xc]
    FADD dword ptr [ESP + 0x4]
    RET
