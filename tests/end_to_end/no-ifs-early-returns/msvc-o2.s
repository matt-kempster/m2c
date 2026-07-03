.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  8B 00 */	mov eax, dword ptr [eax]
/* 00000006 0006  83 F8 08 */	cmp eax, 8
/* 00000009 0009  74 13 */	je .L0000001E
/* 0000000B 000B  83 F8 0F */	cmp eax, 0xf
/* 0000000E 000E  75 15 */	jne .L00000025
/* 00000010 0010  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000014 0014  8B 08 */	mov ecx, dword ptr [eax]
/* 00000016 0016  83 C1 F1 */	add ecx, -0xf
/* 00000019 0019  89 08 */	mov dword ptr [eax], ecx
/* 0000001B 001B  33 C0 */	xor eax, eax
/* 0000001D 001D  C3 */	ret
.L0000001E:
/* 0000001E 001E  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000022 0022  83 00 08 */	add dword ptr [eax], 8
.L00000025:
/* 00000025 0025  33 C0 */	xor eax, eax
/* 00000027 0027  C3 */	ret
/* 00000028 0028  90 */	nop
/* 00000029 0029  90 */	nop
/* 0000002A 002A  90 */	nop
/* 0000002B 002B  90 */	nop
/* 0000002C 002C  90 */	nop
/* 0000002D 002D  90 */	nop
/* 0000002E 002E  90 */	nop
/* 0000002F 002F  90 */	nop

