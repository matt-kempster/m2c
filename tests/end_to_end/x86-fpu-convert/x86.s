# x87 int<->float conversions: real MSVC6 /O2 /QIfist output of
#     int g_i;
#     void to_int(int n, float scale) { g_i = (int)((float)n * scale); }
# fild converts the int argument to float; the /QIfist idiom converts back
# with `fistp qword` into a stack temp, then reads the low dword (int punning
# through the stack, handled by the weak-stack-slot typing). Plain MSVC6 emits
# `call __ftol` for float->int; /QIfist inlines fistp, matching the corpus's
# fistp form (which never calls __ftol).
test:
    SUB ESP, 0x8
    FILD dword ptr [ESP + 0xc]
    FMUL dword ptr [ESP + 0x10]
    FISTP qword ptr [ESP]
    MOV EAX, dword ptr [ESP]
    MOV dword ptr [_g_i], EAX
    ADD ESP, 0x8
    RET
