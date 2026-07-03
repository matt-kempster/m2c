.section .text
test:
/* 00000000 0000  A1 00 00 00 00 */	mov eax, dword ptr [_glob]
/* 00000005 0005  8B C8 */	mov ecx, eax
/* 00000007 0007  48 */	dec eax
/* 00000008 0008  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 0000000D 000D  83 F9 01 */	cmp ecx, 1
/* 00000010 0010  B8 04 00 00 00 */	mov eax, 4
/* 00000015 0015  7C 05 */	jl .L0000001C
/* 00000017 0017  B8 06 00 00 00 */	mov eax, 6
.L0000001C:
/* 0000001C 001C  C3 */	ret
/* 0000001D 001D  90 */	nop
/* 0000001E 001E  90 */	nop
/* 0000001F 001F  90 */	nop

