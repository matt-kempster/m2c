.section .text
frob:
/* 00000000 0000  C3 */	ret

test:
/* 00000010 0000  83 EC 1C */	sub esp, 0x1c
/* 00000013 0003  53 */	push ebx
/* 00000014 0004  55 */	push ebp
/* 00000015 0005  56 */	push esi
/* 00000016 0006  8D 44 24 0F */	lea eax, [esp + 0xf]
/* 0000001A 000A  57 */	push edi
/* 0000001B 000B  50 */	push eax
/* 0000001C 000C  E8 00 00 00 00 */	call frob
/* 00000021 0011  8D 4C 24 1C */	lea ecx, [esp + 0x1c]
/* 00000025 0015  51 */	push ecx
/* 00000026 0016  E8 00 00 00 00 */	call frob
/* 0000002B 001B  8D 54 24 24 */	lea edx, [esp + 0x24]
/* 0000002F 001F  52 */	push edx
/* 00000030 0020  E8 00 00 00 00 */	call frob
/* 00000035 0025  8D 44 24 20 */	lea eax, [esp + 0x20]
/* 00000039 0029  50 */	push eax
/* 0000003A 002A  E8 00 00 00 00 */	call frob
/* 0000003F 002F  8D 4C 24 30 */	lea ecx, [esp + 0x30]
/* 00000043 0033  51 */	push ecx
/* 00000044 0034  E8 00 00 00 00 */	call frob
/* 00000049 0039  8B 44 24 44 */	mov eax, dword ptr [esp + 0x44]
/* 0000004D 003D  83 C4 14 */	add esp, 0x14
/* 00000050 0040  8A 08 */	mov cl, byte ptr [eax]
/* 00000052 0042  8A 58 04 */	mov bl, byte ptr [eax + 4]
/* 00000055 0045  02 CB */	add cl, bl
/* 00000057 0047  66 8B 70 08 */	mov si, word ptr [eax + 8]
/* 0000005B 004B  0F BE E9 */	movsx ebp, cl
/* 0000005E 004E  89 6C 24 30 */	mov dword ptr [esp + 0x30], ebp
/* 00000062 0052  0F AF 28 */	imul ebp, dword ptr [eax]
/* 00000065 0055  66 03 30 */	add si, word ptr [eax]
/* 00000068 0058  8B 50 08 */	mov edx, dword ptr [eax + 8]
/* 0000006B 005B  8B 78 04 */	mov edi, dword ptr [eax + 4]
/* 0000006E 005E  89 6C 24 20 */	mov dword ptr [esp + 0x20], ebp
/* 00000072 0062  0F BF EE */	movsx ebp, si
/* 00000075 0065  89 74 24 18 */	mov dword ptr [esp + 0x18], esi
/* 00000079 0069  8D 1C 17 */	lea ebx, [edi + edx]
/* 0000007C 006C  8B F5 */	mov esi, ebp
/* 0000007E 006E  0F AF D3 */	imul edx, ebx
/* 00000081 0071  0F AF F7 */	imul esi, edi
/* 00000084 0074  84 C9 */	test cl, cl
/* 00000086 0076  88 4C 24 13 */	mov byte ptr [esp + 0x13], cl
/* 0000008A 007A  89 5C 24 1C */	mov dword ptr [esp + 0x1c], ebx
/* 0000008E 007E  89 74 24 24 */	mov dword ptr [esp + 0x24], esi
/* 00000092 0082  89 54 24 28 */	mov dword ptr [esp + 0x28], edx
/* 00000096 0086  75 04 */	jne .L0000009C
/* 00000098 0088  8D 44 24 20 */	lea eax, [esp + 0x20]
.L0000009C:
/* 0000009C 008C  8B 7C 24 30 */	mov edi, dword ptr [esp + 0x30]
/* 000000A0 0090  89 44 24 14 */	mov dword ptr [esp + 0x14], eax
/* 000000A4 0094  8B 00 */	mov eax, dword ptr [eax]
/* 000000A6 0096  03 C5 */	add eax, ebp
/* 000000A8 0098  03 C7 */	add eax, edi
/* 000000AA 009A  5F */	pop edi
/* 000000AB 009B  03 C6 */	add eax, esi
/* 000000AD 009D  5E */	pop esi
/* 000000AE 009E  03 C3 */	add eax, ebx
/* 000000B0 00A0  5D */	pop ebp
/* 000000B1 00A1  5B */	pop ebx
/* 000000B2 00A2  83 C4 1C */	add esp, 0x1c
/* 000000B5 00A5  C3 */	ret

