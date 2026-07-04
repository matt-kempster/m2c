.section .text
func_00400090:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  C3 */	ret

test:
/* 00000010 0000  83 EC 08 */	sub esp, 8
/* 00000013 0003  8B 44 24 0C */	mov eax, dword ptr [esp + 0xc]
/* 00000017 0007  56 */	push esi
/* 00000018 0008  8A 08 */	mov cl, byte ptr [eax]
/* 0000001A 000A  8D 50 28 */	lea edx, [eax + 0x28]
/* 0000001D 000D  8B 80 90 01 00 00 */	mov eax, dword ptr [eax + 0x190]
/* 00000023 0013  88 4C 24 10 */	mov byte ptr [esp + 0x10], cl
/* 00000027 0017  8D 4C 24 10 */	lea ecx, [esp + 0x10]
/* 0000002B 001B  89 54 24 08 */	mov dword ptr [esp + 8], edx
/* 0000002F 001F  51 */	push ecx
/* 00000030 0020  89 44 24 08 */	mov dword ptr [esp + 8], eax
/* 00000034 0024  E8 00 00 00 00 */	call func_00400090
/* 00000039 0029  8A 10 */	mov dl, byte ptr [eax]
/* 0000003B 002B  8A 44 24 14 */	mov al, byte ptr [esp + 0x14]
/* 0000003F 002F  02 C2 */	add al, dl
/* 00000041 0031  88 44 24 14 */	mov byte ptr [esp + 0x14], al
/* 00000045 0035  8D 44 24 0C */	lea eax, [esp + 0xc]
/* 00000049 0039  50 */	push eax
/* 0000004A 003A  E8 00 00 00 00 */	call func_00400090
/* 0000004F 003F  8D 4C 24 0C */	lea ecx, [esp + 0xc]
/* 00000053 0043  89 44 24 10 */	mov dword ptr [esp + 0x10], eax
/* 00000057 0047  51 */	push ecx
/* 00000058 0048  E8 00 00 00 00 */	call func_00400090
/* 0000005D 004D  8B 4C 24 10 */	mov ecx, dword ptr [esp + 0x10]
/* 00000061 0051  8B 10 */	mov edx, dword ptr [eax]
/* 00000063 0053  0F BE 44 24 1C */	movsx eax, byte ptr [esp + 0x1c]
/* 00000068 0058  03 CA */	add ecx, edx
/* 0000006A 005A  8B 54 24 14 */	mov edx, dword ptr [esp + 0x14]
/* 0000006E 005E  89 4C 24 10 */	mov dword ptr [esp + 0x10], ecx
/* 00000072 0062  83 C4 0C */	add esp, 0xc
/* 00000075 0065  8B 32 */	mov esi, dword ptr [edx]
/* 00000077 0067  03 C6 */	add eax, esi
/* 00000079 0069  5E */	pop esi
/* 0000007A 006A  03 C1 */	add eax, ecx
/* 0000007C 006C  83 C4 08 */	add esp, 8
/* 0000007F 006F  C3 */	ret

