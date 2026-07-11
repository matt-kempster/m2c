.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  99 */	cdq
/* 00000005 0005  F7 7C 24 08 */	idiv dword ptr [esp + 8]
/* 00000009 0009  C3 */	ret
/* 0000000A 000A  90 */	nop
/* 0000000B 000B  90 */	nop
/* 0000000C 000C  90 */	nop
/* 0000000D 000D  90 */	nop
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop
