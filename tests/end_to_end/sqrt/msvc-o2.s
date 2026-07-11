.section .text
test:
/* 00000000 0000  51 */	push ecx
/* 00000001 0001  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000005 0005  BA DF 59 37 5F */	mov edx, 0x5f3759df
/* 0000000A 000A  D9 44 24 08 */	fld dword ptr [esp + 8]
/* 0000000E 000E  D8 0D 00 00 00 00 */	fmul dword ptr [_real_3f000000]
/* 00000014 0014  8B C8 */	mov ecx, eax
/* 00000016 0016  89 44 24 00 */	mov dword ptr [esp], eax
/* 0000001A 001A  D1 F9 */	sar ecx, 1
/* 0000001C 001C  2B D1 */	sub edx, ecx
/* 0000001E 001E  89 54 24 00 */	mov dword ptr [esp], edx
/* 00000022 0022  D8 4C 24 00 */	fmul dword ptr [esp]
/* 00000026 0026  D8 4C 24 00 */	fmul dword ptr [esp]
/* 0000002A 002A  D8 2D 00 00 00 00 */	fsubr dword ptr [_real_3fc00000]
/* 00000030 0030  D8 4C 24 00 */	fmul dword ptr [esp]
/* 00000034 0034  59 */	pop ecx
/* 00000035 0035  C3 */	ret
/* 00000036 0036  90 */	nop
/* 00000037 0037  90 */	nop
/* 00000038 0038  90 */	nop
/* 00000039 0039  90 */	nop
/* 0000003A 003A  90 */	nop
/* 0000003B 003B  90 */	nop
/* 0000003C 003C  90 */	nop
/* 0000003D 003D  90 */	nop
/* 0000003E 003E  90 */	nop
/* 0000003F 003F  90 */	nop

.section .rdata
_real_3fc00000:
	.float 1.5

_real_3f000000:
	.float 0.5
