.section .text
test:
/* 00000000 0000  56 */	push esi
/* 00000001 0001  57 */	push edi
/* 00000002 0002  B9 64 00 00 00 */	mov ecx, 0x64
/* 00000007 0007  BE 00 00 00 00 */	mov esi, _b
/* 0000000C 000C  BF 00 00 00 00 */	mov edi, _a
/* 00000011 0011  F3 A5 */	rep movsd dword ptr es:[edi], dword ptr [esi]
/* 00000013 0013  8B 74 24 10 */	mov esi, dword ptr [esp + 0x10]
/* 00000017 0017  8B 7C 24 0C */	mov edi, dword ptr [esp + 0xc]
/* 0000001B 001B  B9 19 00 00 00 */	mov ecx, 0x19
/* 00000020 0020  F3 A5 */	rep movsd dword ptr es:[edi], dword ptr [esi]
/* 00000022 0022  5F */	pop edi
/* 00000023 0023  5E */	pop esi
/* 00000024 0024  C3 */	ret
/* 00000025 0025  90 */	nop
/* 00000026 0026  90 */	nop
/* 00000027 0027  90 */	nop
/* 00000028 0028  90 */	nop
/* 00000029 0029  90 */	nop
/* 0000002A 002A  90 */	nop
/* 0000002B 002B  90 */	nop
/* 0000002C 002C  90 */	nop
/* 0000002D 002D  90 */	nop
/* 0000002E 002E  90 */	nop
/* 0000002F 002F  90 */	nop

