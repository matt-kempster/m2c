.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  33 C0 */	xor eax, eax
/* 00000006 0006  85 C9 */	test ecx, ecx
/* 00000008 0008  74 05 */	je .L0000000F
/* 0000000A 000A  B8 01 00 00 00 */	mov eax, 1
.L0000000F:
/* 0000000F 000F  C3 */	ret

