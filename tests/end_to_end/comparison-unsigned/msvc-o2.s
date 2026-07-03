.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000008 0008  3B C1 */	cmp eax, ecx
/* 0000000A 000A  1B D2 */	sbb edx, edx
/* 0000000C 000C  F7 DA */	neg edx
/* 0000000E 000E  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 00000014 0014  3B C8 */	cmp ecx, eax
/* 00000016 0016  1B C0 */	sbb eax, eax
/* 00000018 0018  40 */	inc eax
/* 00000019 0019  A3 00 00 00 00 */	mov dword ptr [_global], eax
/* 0000001E 001E  C3 */	ret
/* 0000001F 001F  90 */	nop

