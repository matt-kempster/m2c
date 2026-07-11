.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  50 */	push eax
/* 00000005 0005  E8 00 00 00 00 */	call _fabsf
/* 0000000A 000A  D9 1C 24 */	fstp dword ptr [esp]
/* 0000000D 000D  E8 00 00 00 00 */	call _sqrtf
/* 00000012 0012  D9 E1 */	fabs
/* 00000014 0014  83 C4 04 */	add esp, 4
/* 00000017 0017  D9 FA */	fsqrt
/* 00000019 0019  C3 */	ret
/* 0000001A 001A  90 */	nop
/* 0000001B 001B  90 */	nop
/* 0000001C 001C  90 */	nop
/* 0000001D 001D  90 */	nop
/* 0000001E 001E  90 */	nop
/* 0000001F 001F  90 */	nop
