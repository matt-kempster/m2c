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
    FLDL2E
    FSTP dword ptr [_g_l2e]
    FLDL2T
    FSTP dword ptr [_g_l2t]
    FLDLG2
    FSTP dword ptr [_g_lg2]
    FLDLN2
    FSTP dword ptr [_g_ln2]
    FLD dword ptr [EBP + 0x8]
    FLD1
    FPATAN
    FSTP dword ptr [_g_atan]
    FLD dword ptr [EBP + 0x8]
    FLD dword ptr [EBP + 0x8]
    FYL2X
    FSTP dword ptr [_g_logmul]
    FLD dword ptr [EBP + 0x8]
    FLD dword ptr [EBP + 0x8]
    FYL2XP1
    FSTP dword ptr [_g_log1pmul]
    FLD1
    FLD dword ptr [EBP + 0x8]
    FPREM
    FSTP dword ptr [_g_rem]
    FSTP ST(0)
    FLD1
    FLD dword ptr [EBP + 0x8]
    FPREM1
    FSTP dword ptr [_g_rem1]
    FSTP ST(0)
    FLD dword ptr [EBP + 0x8]
    FLD dword ptr [EBP + 0x8]
    FSCALE
    FSTP dword ptr [_g_scale]
    FSTP ST(0)
    FLD dword ptr [EBP + 0x8]
    F2XM1
    FSTP dword ptr [_g_exp2m1]
    FLD dword ptr [EBP + 0x8]
    FRNDINT
    FSTP dword ptr [_g_round]
    POP EBP
    RET
