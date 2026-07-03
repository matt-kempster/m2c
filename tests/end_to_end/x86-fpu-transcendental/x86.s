# x87 transcendentals: real MSVC6 /O2 output of an inline-asm helper
#     float g_wave;
#     void store_wave(float t) { _asm { fld t } _asm { fsin } _asm { fabs }
#                                _asm { fstp g_wave } }
# fsin/fabs map to the libm names sinf/fabsf (MIPS precedent). fpatan/fyl2x/
# fscale/f2xm1 map similarly; frndint stays M2C_RNDINT (control-word bound).
test:
    PUSH EBP
    MOV EBP, ESP
    FLD dword ptr [EBP + 0x8]
    FSIN
    FABS
    FSTP dword ptr [_g_wave]
    POP EBP
    RET
