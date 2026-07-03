.section .text
test:
/* 00000000 0000  51 */	push ecx
/* 00000001 0001  A1 00 00 00 00 */	mov eax, dword ptr [_x]
/* 00000006 0006  D9 05 00 00 00 00 */	fld dword ptr [_real_40a00000]
/* 0000000C 000C  89 44 24 00 */	mov dword ptr [esp], eax
/* 00000010 0010  D9 44 24 00 */	fld dword ptr [esp]
/* 00000014 0014  D8 1D 00 00 00 00 */	fcomp dword ptr [_real_00000000]
/* 0000001A 001A  DF E0 */	fnstsw ax
/* 0000001C 001C  F6 C4 01 */	test ah, 1
/* 0000001F 001F  74 08 */	je .L00000029
/* 00000021 0021  DD D8 */	fstp st(0)
/* 00000023 0023  D9 05 00 00 00 00 */	fld dword ptr [_real_40c00000]
.L00000029:
/* 00000029 0029  D8 1D 00 00 00 00 */	fcomp dword ptr [_real_00000000]
/* 0000002F 002F  C7 05 00 00 00 00 00 00 40 40 */	mov dword ptr [_x], 0x40400000
/* 00000039 0039  DF E0 */	fnstsw ax
/* 0000003B 003B  F6 C4 01 */	test ah, 1
/* 0000003E 003E  75 0A */	jne .L0000004A
/* 00000040 0040  C7 05 00 00 00 00 00 00 E0 40 */	mov dword ptr [_x], 0x40e00000
.L0000004A:
/* 0000004A 004A  59 */	pop ecx
/* 0000004B 004B  C3 */	ret
/* 0000004C 004C  90 */	nop
/* 0000004D 004D  90 */	nop
/* 0000004E 004E  90 */	nop
/* 0000004F 004F  90 */	nop

.section .rdata
_real_40c00000:
	.float 6.0

_real_00000000:
	.float 0.0

_real_40a00000:
	.float 5.0

