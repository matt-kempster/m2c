.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  83 F9 05 */	cmp ecx, 5
/* 00000007 0007  75 3D */	jne .L00000046
.L00000009:
/* 00000009 0009  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 0000000F 000F  41 */	inc ecx
/* 00000010 0010  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 00000016 0016  41 */	inc ecx
/* 00000017 0017  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 0000001D 001D  41 */	inc ecx
/* 0000001E 001E  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 00000024 0024  8B C1 */	mov eax, ecx
/* 00000026 0026  41 */	inc ecx
/* 00000027 0027  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 0000002D 002D  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 00000033 0033  41 */	inc ecx
/* 00000034 0034  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 0000003A 003A  41 */	inc ecx
/* 0000003B 003B  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 00000040 0040  83 F9 05 */	cmp ecx, 5
/* 00000043 0043  74 C4 */	je .L00000009
/* 00000045 0045  C3 */	ret
.L00000046:
/* 00000046 0046  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 0000004A 004A  C3 */	ret
/* 0000004B 004B  90 */	nop
/* 0000004C 004C  90 */	nop
/* 0000004D 004D  90 */	nop
/* 0000004E 004E  90 */	nop
/* 0000004F 004F  90 */	nop

