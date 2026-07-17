g_base:
    .double 1.25
g_double:
    .double 2.5

test:
    FLD qword ptr [g_base]
    FADD dword ptr g_double
    RET
