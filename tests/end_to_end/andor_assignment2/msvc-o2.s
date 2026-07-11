.section .text
foo:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  40 */	inc eax
/* 00000005 0005  C3 */	ret
/* 00000006 0006  90 */	nop
/* 00000007 0007  90 */	nop
/* 00000008 0008  90 */	nop
/* 00000009 0009  90 */	nop
/* 0000000A 000A  90 */	nop
/* 0000000B 000B  90 */	nop
/* 0000000C 000C  90 */	nop
/* 0000000D 000D  90 */	nop
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop

test:
/* 00000010 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000014 0004  8B 54 24 10 */	mov edx, dword ptr [esp + 0x10]
/* 00000018 0008  53 */	push ebx
/* 00000019 0009  55 */	push ebp
/* 0000001A 000A  56 */	push esi
/* 0000001B 000B  8B 74 24 14 */	mov esi, dword ptr [esp + 0x14]
/* 0000001F 000F  57 */	push edi
/* 00000020 0010  8B 7C 24 1C */	mov edi, dword ptr [esp + 0x1c]
/* 00000024 0014  8D 04 31 */	lea eax, [ecx + esi]
/* 00000027 0017  85 C0 */	test eax, eax
/* 00000029 0019  8D 1C 3E */	lea ebx, [esi + edi]
/* 0000002C 001C  8D 2C 17 */	lea ebp, [edi + edx]
/* 0000002F 001F  74 51 */	je .L00000082
/* 00000031 0021  85 DB */	test ebx, ebx
/* 00000033 0023  74 4D */	je .L00000082
/* 00000035 0025  85 ED */	test ebp, ebp
/* 00000037 0027  74 49 */	je .L00000082
/* 00000039 0029  03 C1 */	add eax, ecx
/* 0000003B 002B  50 */	push eax
/* 0000003C 002C  E8 00 00 00 00 */	call foo
/* 00000041 0031  83 C4 04 */	add esp, 4
/* 00000044 0034  83 F8 0A */	cmp eax, 0xa
/* 00000047 0037  7E 39 */	jle .L00000082
/* 00000049 0039  03 C6 */	add eax, esi
/* 0000004B 003B  50 */	push eax
/* 0000004C 003C  E8 00 00 00 00 */	call foo
/* 00000051 0041  03 DF */	add ebx, edi
/* 00000053 0043  8B F0 */	mov esi, eax
/* 00000055 0045  53 */	push ebx
/* 00000056 0046  E8 00 00 00 00 */	call foo
/* 0000005B 004B  8B F8 */	mov edi, eax
/* 0000005D 004D  8B 44 24 28 */	mov eax, dword ptr [esp + 0x28]
/* 00000061 0051  03 E8 */	add ebp, eax
/* 00000063 0053  55 */	push ebp
/* 00000064 0054  E8 00 00 00 00 */	call foo
/* 00000069 0059  83 C4 0C */	add esp, 0xc
/* 0000006C 005C  85 F6 */	test esi, esi
/* 0000006E 005E  74 12 */	je .L00000082
/* 00000070 0060  85 FF */	test edi, edi
/* 00000072 0062  74 0E */	je .L00000082
/* 00000074 0064  85 C0 */	test eax, eax
/* 00000076 0066  74 0A */	je .L00000082
/* 00000078 0068  5F */	pop edi
/* 00000079 0069  5E */	pop esi
/* 0000007A 006A  5D */	pop ebp
/* 0000007B 006B  B8 01 00 00 00 */	mov eax, 1
/* 00000080 0070  5B */	pop ebx
/* 00000081 0071  C3 */	ret
.L00000082:
/* 00000082 0072  5F */	pop edi
/* 00000083 0073  5E */	pop esi
/* 00000084 0074  5D */	pop ebp
/* 00000085 0075  33 C0 */	xor eax, eax
/* 00000087 0077  5B */	pop ebx
/* 00000088 0078  C3 */	ret
/* 00000089 0079  90 */	nop
/* 0000008A 007A  90 */	nop
/* 0000008B 007B  90 */	nop
/* 0000008C 007C  90 */	nop
/* 0000008D 007D  90 */	nop
/* 0000008E 007E  90 */	nop
/* 0000008F 007F  90 */	nop
