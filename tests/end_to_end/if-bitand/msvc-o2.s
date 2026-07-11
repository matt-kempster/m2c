.section .text
test:
/* 00000000 0000  33 C0 */	xor eax, eax
/* 00000002 0002  B1 01 */	mov cl, 1
/* 00000004 0004  84 0D 00 00 00 00 */	test byte ptr [_glob], cl
/* 0000000A 000A  74 05 */	je .L00000011
/* 0000000C 000C  A3 00 00 00 00 */	mov dword ptr [_glob], eax
.L00000011:
/* 00000011 0011  F7 05 00 00 00 00 00 00 01 00 */	test dword ptr [_glob], 0x10000
/* 0000001B 001B  74 05 */	je .L00000022
/* 0000001D 001D  A3 00 00 00 00 */	mov dword ptr [_glob], eax
.L00000022:
/* 00000022 0022  BA 00 00 00 80 */	mov edx, 0x80000000
/* 00000027 0027  85 15 00 00 00 00 */	test dword ptr [_glob], edx
/* 0000002D 002D  74 05 */	je .L00000034
/* 0000002F 002F  A3 00 00 00 00 */	mov dword ptr [_glob], eax
.L00000034:
/* 00000034 0034  53 */	push ebx
/* 00000035 0035  8A 1D 00 00 00 00 */	mov bl, byte ptr [_glob]
/* 0000003B 003B  84 D9 */	test cl, bl
/* 0000003D 003D  5B */	pop ebx
/* 0000003E 003E  74 05 */	je .L00000045
/* 00000040 0040  A3 00 00 00 00 */	mov dword ptr [_glob], eax
.L00000045:
/* 00000045 0045  F7 05 00 00 00 00 00 00 01 00 */	test dword ptr [_glob], 0x10000
/* 0000004F 004F  74 05 */	je .L00000056
/* 00000051 0051  A3 00 00 00 00 */	mov dword ptr [_glob], eax
.L00000056:
/* 00000056 0056  85 15 00 00 00 00 */	test dword ptr [_glob], edx
/* 0000005C 005C  74 05 */	je .L00000063
/* 0000005E 005E  A3 00 00 00 00 */	mov dword ptr [_glob], eax
.L00000063:
/* 00000063 0063  C3 */	ret
