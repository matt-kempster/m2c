.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  8B 08 */	mov ecx, dword ptr [eax]
/* 00000006 0006  8B 50 04 */	mov edx, dword ptr [eax + 4]
/* 00000009 0009  03 D1 */	add edx, ecx
/* 0000000B 000B  89 50 04 */	mov dword ptr [eax + 4], edx
/* 0000000E 000E  8B D1 */	mov edx, ecx
/* 00000010 0010  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000014 0014  89 11 */	mov dword ptr [ecx], edx
/* 00000016 0016  8B 50 04 */	mov edx, dword ptr [eax + 4]
/* 00000019 0019  89 51 04 */	mov dword ptr [ecx + 4], edx
/* 0000001C 001C  C3 */	ret
/* 0000001D 001D  90 */	nop
/* 0000001E 001E  90 */	nop
/* 0000001F 001F  90 */	nop
