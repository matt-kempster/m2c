.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  55 */	push ebp
/* 00000005 0005  33 C0 */	xor eax, eax
/* 00000007 0007  56 */	push esi
/* 00000008 0008  85 C9 */	test ecx, ecx
/* 0000000A 000A  57 */	push edi
/* 0000000B 000B  0F 8E 8D 00 00 00 */	jle .L0000009E
/* 00000011 0011  8B 15 14 00 00 00 */	mov edx, dword ptr [_globals + 0x14]
/* 00000017 0017  8B 35 10 00 00 00 */	mov esi, dword ptr [_globals + 0x10]
/* 0000001D 001D  8B 3D 08 00 00 00 */	mov edi, dword ptr [_globals + 0x8]
/* 00000023 0023  8B 0D 04 00 00 00 */	mov ecx, dword ptr [_globals + 0x4]
/* 00000029 0029  BD 03 00 00 00 */	mov ebp, 3
/* 0000002E 002E  EB 05 */	jmp .L00000035
.L00000030:
/* 00000030 0030  BD 03 00 00 00 */	mov ebp, 3
.L00000035:
/* 00000035 0035  83 F9 02 */	cmp ecx, 2
/* 00000038 0038  C7 05 00 00 00 00 01 00 00 00 */	mov dword ptr [_globals], 1
/* 00000042 0042  74 40 */	je .L00000084
/* 00000044 0044  83 FF 02 */	cmp edi, 2
/* 00000047 0047  75 08 */	jne .L00000051
/* 00000049 0049  89 2D 0C 00 00 00 */	mov dword ptr [_globals + 0xc], ebp
/* 0000004F 004F  EB 1C */	jmp .L0000006D
.L00000051:
/* 00000051 0051  83 FE 02 */	cmp esi, 2
/* 00000054 0054  74 42 */	je .L00000098
/* 00000056 0056  83 FA 02 */	cmp edx, 2
/* 00000059 0059  75 08 */	jne .L00000063
/* 0000005B 005B  89 2D 18 00 00 00 */	mov dword ptr [_globals + 0x18], ebp
/* 00000061 0061  EB 0A */	jmp .L0000006D
.L00000063:
/* 00000063 0063  C7 05 0C 00 00 00 04 00 00 00 */	mov dword ptr [_globals + 0xc], 4
.L0000006D:
/* 0000006D 006D  8B 6C 24 10 */	mov ebp, dword ptr [esp + 0x10]
/* 00000071 0071  40 */	inc eax
/* 00000072 0072  3B C5 */	cmp eax, ebp
/* 00000074 0074  7C BA */	jl .L00000030
/* 00000076 0076  5F */	pop edi
/* 00000077 0077  5E */	pop esi
/* 00000078 0078  C7 05 10 00 00 00 05 00 00 00 */	mov dword ptr [_globals + 0x10], 5
/* 00000082 0082  5D */	pop ebp
/* 00000083 0083  C3 */	ret
.L00000084:
/* 00000084 0084  5F */	pop edi
/* 00000085 0085  89 2D 08 00 00 00 */	mov dword ptr [_globals + 0x8], ebp
/* 0000008B 008B  5E */	pop esi
/* 0000008C 008C  C7 05 10 00 00 00 05 00 00 00 */	mov dword ptr [_globals + 0x10], 5
/* 00000096 0096  5D */	pop ebp
/* 00000097 0097  C3 */	ret
.L00000098:
/* 00000098 0098  89 2D 14 00 00 00 */	mov dword ptr [_globals + 0x14], ebp
.L0000009E:
/* 0000009E 009E  5F */	pop edi
/* 0000009F 009F  5E */	pop esi
/* 000000A0 00A0  C7 05 10 00 00 00 05 00 00 00 */	mov dword ptr [_globals + 0x10], 5
/* 000000AA 00AA  5D */	pop ebp
/* 000000AB 00AB  C3 */	ret
/* 000000AC 00AC  90 */	nop
/* 000000AD 00AD  90 */	nop
/* 000000AE 00AE  90 */	nop
/* 000000AF 00AF  90 */	nop
