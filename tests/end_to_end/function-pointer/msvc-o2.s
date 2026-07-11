.section .text
test:
/* 00000000 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000004 0004  50 */	push eax
/* 00000005 0005  FF 54 24 08 */	call dword ptr [esp + 8]
/* 00000009 0009  83 C4 04 */	add esp, 4
/* 0000000C 000C  40 */	inc eax
/* 0000000D 000D  C3 */	ret
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop
