.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000008 0008  33 D2 */	xor edx, edx
/* 0000000A 000A  3B C8 */	cmp ecx, eax
/* 0000000C 000C  0F 94 C2 */	sete dl
/* 0000000F 000F  56 */	push esi
/* 00000010 0010  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 00000016 0016  8B 74 24 10 */	mov esi, dword ptr [esp + 0x10]
/* 0000001A 001A  33 D2 */	xor edx, edx
/* 0000001C 001C  3B CE */	cmp ecx, esi
/* 0000001E 001E  0F 95 C2 */	setne dl
/* 00000021 0021  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 00000027 0027  33 D2 */	xor edx, edx
/* 00000029 0029  3B C8 */	cmp ecx, eax
/* 0000002B 002B  0F 9C C2 */	setl dl
/* 0000002E 002E  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 00000034 0034  33 D2 */	xor edx, edx
/* 00000036 0036  3B C8 */	cmp ecx, eax
/* 00000038 0038  0F 9E C2 */	setle dl
/* 0000003B 003B  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 00000041 0041  33 D2 */	xor edx, edx
/* 00000043 0043  85 C9 */	test ecx, ecx
/* 00000045 0045  0F 94 C2 */	sete dl
/* 00000048 0048  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 0000004E 004E  33 C9 */	xor ecx, ecx
/* 00000050 0050  85 C0 */	test eax, eax
/* 00000052 0052  0F 95 C1 */	setne cl
/* 00000055 0055  89 0D 00 00 00 00 */	mov dword ptr [_global], ecx
/* 0000005B 005B  33 D2 */	xor edx, edx
/* 0000005D 005D  85 C0 */	test eax, eax
/* 0000005F 005F  0F 9F C2 */	setg dl
/* 00000062 0062  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 00000068 0068  33 C9 */	xor ecx, ecx
/* 0000006A 006A  85 C0 */	test eax, eax
/* 0000006C 006C  0F 9E C1 */	setle cl
/* 0000006F 006F  89 0D 00 00 00 00 */	mov dword ptr [_global], ecx
/* 00000075 0075  5E */	pop esi
/* 00000076 0076  C3 */	ret
/* 00000077 0077  90 */	nop
/* 00000078 0078  90 */	nop
/* 00000079 0079  90 */	nop
/* 0000007A 007A  90 */	nop
/* 0000007B 007B  90 */	nop
/* 0000007C 007C  90 */	nop
/* 0000007D 007D  90 */	nop
/* 0000007E 007E  90 */	nop
/* 0000007F 007F  90 */	nop
