.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  8D 48 FF */	lea ecx, [eax - 1]
/* 00000007 0007  83 F9 06 */	cmp ecx, 6
/* 0000000A 000A  77 2C */	ja $L100
/* 0000000C 000C  FF 24 8D 00 00 00 00 */	jmp dword ptr [ecx*4 + $L105]
$L95:
/* 00000013 0013  0F AF C0 */	imul eax, eax
/* 00000016 0016  C3 */	ret
$L96:
/* 00000017 0017  B8 01 00 00 00 */	mov eax, 1
$L97:
/* 0000001C 001C  03 C0 */	add eax, eax
/* 0000001E 001E  C3 */	ret
$L98:
/* 0000001F 001F  40 */	inc eax
/* 00000020 0020  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 00000025 0025  B8 02 00 00 00 */	mov eax, 2
/* 0000002A 002A  C3 */	ret
$L99:
/* 0000002B 002B  03 C0 */	add eax, eax
/* 0000002D 002D  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 00000032 0032  B8 02 00 00 00 */	mov eax, 2
/* 00000037 0037  C3 */	ret
$L100:
/* 00000038 0038  99 */	cdq
/* 00000039 0039  2B C2 */	sub eax, edx
/* 0000003B 003B  D1 F8 */	sar eax, 1
/* 0000003D 003D  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 00000042 0042  B8 02 00 00 00 */	mov eax, 2
/* 00000047 0047  C3 */	ret
$L105:
	.long $L95
	.long $L96
	.long $L97
	.long $L98
	.long $L100
	.long $L99
	.long $L99

