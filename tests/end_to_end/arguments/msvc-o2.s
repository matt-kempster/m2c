.section .text
test:
/* 00000000 0000  D9 44 24 04 */	fld dword ptr [esp + 4]
/* 00000004 0004  D8 44 24 0C */	fadd dword ptr [esp + 0xc]
/* 00000008 0008  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 0000000C 000C  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000010 0010  03 C8 */	add ecx, eax
/* 00000012 0012  8B 44 24 18 */	mov eax, dword ptr [esp + 0x18]
/* 00000016 0016  D8 44 24 14 */	fadd dword ptr [esp + 0x14]
/* 0000001A 001A  03 C8 */	add ecx, eax
/* 0000001C 001C  89 0D 00 00 00 00 */	mov dword ptr [_globali], ecx
/* 00000022 0022  D9 1D 00 00 00 00 */	fstp dword ptr [_globalf]
/* 00000028 0028  C3 */	ret
/* 00000029 0029  90 */	nop
/* 0000002A 002A  90 */	nop
/* 0000002B 002B  90 */	nop
/* 0000002C 002C  90 */	nop
/* 0000002D 002D  90 */	nop
/* 0000002E 002E  90 */	nop
/* 0000002F 002F  90 */	nop
