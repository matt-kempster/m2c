.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  33 C0 */	xor eax, eax
/* 00000006 0006  85 C9 */	test ecx, ecx
/* 00000008 0008  0F 95 C0 */	setne al
/* 0000000B 000B  C3 */	ret
/* 0000000C 000C  90 */	nop
/* 0000000D 000D  90 */	nop
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop
