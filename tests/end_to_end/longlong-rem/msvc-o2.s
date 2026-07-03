.section .text
test:
/* 00000000 0000  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 00000004 0004  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 00000008 0008  8B 54 24 08 */	mov edx, dword ptr [esp + 8]
/* 0000000C 000C  50 */	push eax
/* 0000000D 000D  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000011 0011  51 */	push ecx
/* 00000012 0012  52 */	push edx
/* 00000013 0013  50 */	push eax
/* 00000014 0014  E8 00 00 00 00 */	call __allrem
/* 00000019 0019  C3 */	ret
/* 0000001A 001A  90 */	nop
/* 0000001B 001B  90 */	nop
/* 0000001C 001C  90 */	nop
/* 0000001D 001D  90 */	nop
/* 0000001E 001E  90 */	nop
/* 0000001F 001F  90 */	nop

