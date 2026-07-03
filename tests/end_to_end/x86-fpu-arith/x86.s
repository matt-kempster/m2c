# x87 arithmetic: real MSVC6 /O2 output of
#     float g_result;
#     void store_expr(float a, float b, float c) {
#         g_result = (a + b) * c - a / b;
#     }
# Exercises fld/fadd/fmul/fdiv, the popping fsubp accumulator combine, and a
# store-and-pop back to a global (no float return, which slice 4 handles).
test:
    FLD dword ptr [ESP + 0x4]
    FADD dword ptr [ESP + 0x8]
    FMUL dword ptr [ESP + 0xc]
    FLD dword ptr [ESP + 0x4]
    FDIV dword ptr [ESP + 0x8]
    FSUBP st(1), st(0)
    FSTP dword ptr [_g_result]
    RET
