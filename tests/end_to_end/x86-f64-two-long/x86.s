test:
    FLD qword ptr [_two_long]
    RET

.section .rodata
_two_long:
    .long 0x00000000
    .long 0x401c0000
