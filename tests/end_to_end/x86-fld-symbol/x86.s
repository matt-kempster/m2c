# Some Intel-syntax inputs write a direct memory operand without brackets when
# it is a bare
# symbol (`fld _sym`, `fadd _sym+8`), unlike the bracketed `[esp+N]` forms.
# These are absolute memory loads, not address immediates.
g_scale:
    .4byte 0x40490fdb

test:
    FLD g_scale
    FMUL dword ptr [ESP + 0x4]
    RET
