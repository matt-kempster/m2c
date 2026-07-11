.section .text
foo:
/* 00000000 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000004 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000008 0008  8B 00 */	mov eax, dword ptr [eax]
/* 0000000A 000A  03 C1 */	add eax, ecx
/* 0000000C 000C  C3 */	ret
/* 0000000D 000D  90 */	nop
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop

test:
/* 00000010 0000  56 */	push esi
/* 00000011 0001  8B 74 24 0C */	mov esi, dword ptr [esp + 0xc]
/* 00000015 0005  57 */	push edi
/* 00000016 0006  56 */	push esi
/* 00000017 0007  8B 06 */	mov eax, dword ptr [esi]
/* 00000019 0009  50 */	push eax
/* 0000001A 000A  E8 00 00 00 00 */	call foo
/* 0000001F 000F  8B 4C 24 14 */	mov ecx, dword ptr [esp + 0x14]
/* 00000023 0013  56 */	push esi
/* 00000024 0014  51 */	push ecx
/* 00000025 0015  8B F8 */	mov edi, eax
/* 00000027 0017  E8 00 00 00 00 */	call foo
/* 0000002C 001C  03 F8 */	add edi, eax
/* 0000002E 001E  8B 44 24 1C */	mov eax, dword ptr [esp + 0x1c]
/* 00000032 0022  8D 54 24 1C */	lea edx, [esp + 0x1c]
/* 00000036 0026  52 */	push edx
/* 00000037 0027  50 */	push eax
/* 00000038 0028  E8 00 00 00 00 */	call foo
/* 0000003D 002D  83 C4 18 */	add esp, 0x18
/* 00000040 0030  03 C7 */	add eax, edi
/* 00000042 0032  5F */	pop edi
/* 00000043 0033  5E */	pop esi
/* 00000044 0034  C3 */	ret
/* 00000045 0035  90 */	nop
/* 00000046 0036  90 */	nop
/* 00000047 0037  90 */	nop
/* 00000048 0038  90 */	nop
/* 00000049 0039  90 */	nop
/* 0000004A 003A  90 */	nop
/* 0000004B 003B  90 */	nop
/* 0000004C 003C  90 */	nop
/* 0000004D 003D  90 */	nop
/* 0000004E 003E  90 */	nop
/* 0000004F 003F  90 */	nop
