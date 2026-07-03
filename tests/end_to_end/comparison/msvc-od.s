.section .text
test:
/* 00000000 0000  55 */	push ebp
/* 00000001 0001  8B EC */	mov ebp, esp
/* 00000003 0003  8B 45 08 */	mov eax, dword ptr [ebp + 8]
/* 00000006 0006  33 C9 */	xor ecx, ecx
/* 00000008 0008  3B 45 0C */	cmp eax, dword ptr [ebp + 0xc]
/* 0000000B 000B  0F 94 C1 */	sete cl
/* 0000000E 000E  89 0D 00 00 00 00 */	mov dword ptr [_global], ecx
/* 00000014 0014  8B 55 08 */	mov edx, dword ptr [ebp + 8]
/* 00000017 0017  33 C0 */	xor eax, eax
/* 00000019 0019  3B 55 10 */	cmp edx, dword ptr [ebp + 0x10]
/* 0000001C 001C  0F 95 C0 */	setne al
/* 0000001F 001F  A3 00 00 00 00 */	mov dword ptr [_global], eax
/* 00000024 0024  8B 4D 08 */	mov ecx, dword ptr [ebp + 8]
/* 00000027 0027  33 D2 */	xor edx, edx
/* 00000029 0029  3B 4D 0C */	cmp ecx, dword ptr [ebp + 0xc]
/* 0000002C 002C  0F 9C C2 */	setl dl
/* 0000002F 002F  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 00000035 0035  8B 45 08 */	mov eax, dword ptr [ebp + 8]
/* 00000038 0038  33 C9 */	xor ecx, ecx
/* 0000003A 003A  3B 45 0C */	cmp eax, dword ptr [ebp + 0xc]
/* 0000003D 003D  0F 9E C1 */	setle cl
/* 00000040 0040  89 0D 00 00 00 00 */	mov dword ptr [_global], ecx
/* 00000046 0046  33 D2 */	xor edx, edx
/* 00000048 0048  83 7D 08 00 */	cmp dword ptr [ebp + 8], 0
/* 0000004C 004C  0F 94 C2 */	sete dl
/* 0000004F 004F  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 00000055 0055  33 C0 */	xor eax, eax
/* 00000057 0057  83 7D 0C 00 */	cmp dword ptr [ebp + 0xc], 0
/* 0000005B 005B  0F 95 C0 */	setne al
/* 0000005E 005E  A3 00 00 00 00 */	mov dword ptr [_global], eax
/* 00000063 0063  33 C9 */	xor ecx, ecx
/* 00000065 0065  83 7D 0C 00 */	cmp dword ptr [ebp + 0xc], 0
/* 00000069 0069  0F 9F C1 */	setg cl
/* 0000006C 006C  89 0D 00 00 00 00 */	mov dword ptr [_global], ecx
/* 00000072 0072  33 D2 */	xor edx, edx
/* 00000074 0074  83 7D 0C 00 */	cmp dword ptr [ebp + 0xc], 0
/* 00000078 0078  0F 9E C2 */	setle dl
/* 0000007B 007B  89 15 00 00 00 00 */	mov dword ptr [_global], edx
/* 00000081 0081  5D */	pop ebp
/* 00000082 0082  C3 */	ret

