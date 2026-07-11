.section .text
test:
/* 00000000 0000  D9 44 24 04 */	fld dword ptr [esp + 4]
/* 00000004 0004  D8 1D 00 00 00 00 */	fcomp dword ptr [_real_00000000]
/* 0000000A 000A  DF E0 */	fnstsw ax
/* 0000000C 000C  F6 C4 40 */	test ah, 0x40
/* 0000000F 000F  75 07 */	jne .L00000018
/* 00000011 0011  D9 05 00 00 00 00 */	fld dword ptr [_real_41700000]
/* 00000017 0017  C3 */	ret
.L00000018:
/* 00000018 0018  D9 44 24 04 */	fld dword ptr [esp + 4]
/* 0000001C 001C  C3 */	ret
/* 0000001D 001D  90 */	nop
/* 0000001E 001E  90 */	nop
/* 0000001F 001F  90 */	nop

.section .rdata
_real_41700000:
	.float 15.0

_real_00000000:
	.float 0.0
