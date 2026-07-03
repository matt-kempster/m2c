.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  8B 4C 24 10 */	mov ecx, dword ptr [esp + 0x10]
/* 00000008 0008  8B 54 24 0C */	mov edx, dword ptr [esp + 0xc]
/* 0000000C 000C  56 */	push esi
/* 0000000D 000D  8B 74 24 0C */	mov esi, dword ptr [esp + 0xc]
/* 00000011 0011  85 C0 */	test eax, eax
/* 00000013 0013  75 04 */	jne .L00000019
/* 00000015 0015  85 F6 */	test esi, esi
/* 00000017 0017  74 08 */	je .L00000021
.L00000019:
/* 00000019 0019  85 D2 */	test edx, edx
/* 0000001B 001B  75 09 */	jne .L00000026
/* 0000001D 001D  85 C9 */	test ecx, ecx
/* 0000001F 001F  75 05 */	jne .L00000026
.L00000021:
/* 00000021 0021  8D 04 0A */	lea eax, [edx + ecx]
/* 00000024 0024  5E */	pop esi
/* 00000025 0025  C3 */	ret
.L00000026:
/* 00000026 0026  03 C6 */	add eax, esi
/* 00000028 0028  5E */	pop esi
/* 00000029 0029  C3 */	ret
/* 0000002A 002A  90 */	nop
/* 0000002B 002B  90 */	nop
/* 0000002C 002C  90 */	nop
/* 0000002D 002D  90 */	nop
/* 0000002E 002E  90 */	nop
/* 0000002F 002F  90 */	nop

