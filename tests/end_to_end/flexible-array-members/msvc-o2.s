.section .text
test:
/* 00000000 0000  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 00000004 0004  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000008 0008  8B 54 24 08 */	mov edx, dword ptr [esp + 8]
/* 0000000C 000C  56 */	push esi
/* 0000000D 000D  8B 44 88 04 */	mov eax, dword ptr [eax + ecx*4 + 4]
/* 00000011 0011  8B 74 CA 08 */	mov esi, dword ptr [edx + ecx*8 + 8]
/* 00000015 0015  03 C6 */	add eax, esi
/* 00000017 0017  5E */	pop esi
/* 00000018 0018  C3 */	ret
/* 00000019 0019  90 */	nop
/* 0000001A 001A  90 */	nop
/* 0000001B 001B  90 */	nop
/* 0000001C 001C  90 */	nop
/* 0000001D 001D  90 */	nop
/* 0000001E 001E  90 */	nop
/* 0000001F 001F  90 */	nop

