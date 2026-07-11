.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  8D 41 FF */	lea eax, [ecx - 1]
/* 00000007 0007  83 F8 05 */	cmp eax, 5
/* 0000000A 000A  77 65 */	ja .L00000071
/* 0000000C 000C  FF 24 85 00 00 00 00 */	jmp dword ptr [eax*4 + $L114]
$L95:
/* 00000013 0013  C7 05 00 00 00 00 01 00 00 00 */	mov dword ptr [_glob], 1
/* 0000001D 001D  83 F9 01 */	cmp ecx, 1
/* 00000020 0020  75 1B */	jne .L0000003D
/* 00000022 0022  C7 05 00 00 00 00 02 00 00 00 */	mov dword ptr [_glob], 2
/* 0000002C 002C  EB 0F */	jmp .L0000003D
$L97:
/* 0000002E 002E  33 D2 */	xor edx, edx
/* 00000030 0030  83 F9 01 */	cmp ecx, 1
/* 00000033 0033  0F 95 C2 */	setne dl
/* 00000036 0036  42 */	inc edx
/* 00000037 0037  89 15 00 00 00 00 */	mov dword ptr [_glob], edx
.L0000003D:
/* 0000003D 003D  83 F8 05 */	cmp eax, 5
/* 00000040 0040  77 2F */	ja .L00000071
/* 00000042 0042  FF 24 85 00 00 00 00 */	jmp dword ptr [eax*4 + $L115]
$L104:
/* 00000049 0049  C7 05 00 00 00 00 01 00 00 00 */	mov dword ptr [_glob], 1
/* 00000053 0053  83 F9 01 */	cmp ecx, 1
/* 00000056 0056  75 19 */	jne .L00000071
/* 00000058 0058  C7 05 00 00 00 00 02 00 00 00 */	mov dword ptr [_glob], 2
/* 00000062 0062  C3 */	ret
$L106:
/* 00000063 0063  33 C0 */	xor eax, eax
/* 00000065 0065  83 F9 01 */	cmp ecx, 1
/* 00000068 0068  0F 95 C0 */	setne al
/* 0000006B 006B  40 */	inc eax
/* 0000006C 006C  A3 00 00 00 00 */	mov dword ptr [_glob], eax
.L00000071:
/* 00000071 0071  C3 */	ret
/* 00000072 0072  8B FF */	mov edi, edi
$L114:
	.long $L95
	.long $L95
	.long $L95
	.long $L95
	.long $L97
	.long $L97
$L115:
	.long $L104
	.long $L104
	.long $L104
	.long $L104
	.long $L106
	.long $L106
