.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000008 0008  0B C1 */	or eax, ecx
/* 0000000A 000A  F7 D0 */	not eax
/* 0000000C 000C  C3 */	ret
/* 0000000D 000D  90 */	nop
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop
