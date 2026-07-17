g_base:
    .double 1.25
g_addend:
    .double 2.5

test:
    FLD qword ptr [g_base]
    FADD g_addend
    RET
