.section .text
test:
/* 00000000 0000  8D 44 24 08 */	lea eax, [esp + 8]
/* 00000004 0004  8D 4C 24 04 */	lea ecx, [esp + 4]
/* 00000008 0008  2B C1 */	sub eax, ecx
/* 0000000A 000A  C1 F8 02 */	sar eax, 2
/* 0000000D 000D  C3 */	ret
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop
