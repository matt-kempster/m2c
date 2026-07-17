g_base:
    .double 1.25
g_padded_float:
    .float 2.5
    .space 4

test:
    FLD qword ptr [g_base]
    FADD g_padded_float
    RET
