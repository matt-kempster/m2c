.section .text
test:
/* 00000000 0000  D9 05 00 00 00 00 */	fld dword ptr [_x]
/* 00000006 0006  DC C0 */	fadd st(0), st(0)
/* 00000008 0008  D9 1D 00 00 00 00 */	fstp dword ptr [_x]
/* 0000000E 000E  DD 05 00 00 00 00 */	fld qword ptr [_y]
/* 00000014 0014  DC C0 */	fadd st(0), st(0)
/* 00000016 0016  DD 1D 00 00 00 00 */	fstp qword ptr [_y]
/* 0000001C 001C  C3 */	ret
/* 0000001D 001D  90 */	nop
/* 0000001E 001E  90 */	nop
/* 0000001F 001F  90 */	nop

