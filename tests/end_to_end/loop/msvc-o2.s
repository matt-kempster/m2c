.section .text
test:
/* 00000000 0000  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000004 0004  85 C9 */	test ecx, ecx
/* 00000006 0006  7E 16 */	jle .L0000001E
/* 00000008 0008  8B D1 */	mov edx, ecx
/* 0000000A 000A  57 */	push edi
/* 0000000B 000B  8B 7C 24 08 */	mov edi, dword ptr [esp + 8]
/* 0000000F 000F  33 C0 */	xor eax, eax
/* 00000011 0011  C1 E9 02 */	shr ecx, 2
/* 00000014 0014  F3 AB */	rep stosd dword ptr es:[edi], eax
/* 00000016 0016  8B CA */	mov ecx, edx
/* 00000018 0018  83 E1 03 */	and ecx, 3
/* 0000001B 001B  F3 AA */	rep stosb byte ptr es:[edi], al
/* 0000001D 001D  5F */	pop edi
.L0000001E:
/* 0000001E 001E  C3 */	ret
/* 0000001F 001F  90 */	nop

