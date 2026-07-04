.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  8B 01 */	mov eax, dword ptr [ecx]
/* 00000006 0006  83 E8 00 */	sub eax, 0
/* 00000009 0009  74 0D */	je .L00000018
/* 0000000B 000B  48 */	dec eax
/* 0000000C 000C  74 0A */	je .L00000018
/* 0000000E 000E  48 */	dec eax
/* 0000000F 000F  74 07 */	je .L00000018
/* 00000011 0011  D9 05 00 00 00 00 */	fld dword ptr [_real_00000000]
/* 00000017 0017  C3 */	ret
.L00000018:
/* 00000018 0018  D9 41 0C */	fld dword ptr [ecx + 0xc]
/* 0000001B 001B  D8 41 04 */	fadd dword ptr [ecx + 4]
/* 0000001E 001E  C3 */	ret

.section .rdata
_real_00000000:
	.float 0.0

