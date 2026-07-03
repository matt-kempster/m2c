.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  8B 01 */	mov eax, dword ptr [ecx]
/* 00000006 0006  99 */	cdq
/* 00000007 0007  2B C2 */	sub eax, edx
/* 00000009 0009  D1 F8 */	sar eax, 1
/* 0000000B 000B  89 01 */	mov dword ptr [ecx], eax
/* 0000000D 000D  C3 */	ret
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop

