.section .text
test:
/* 00000000 0000  0F BE 05 00 00 00 00 */	movsx eax, byte ptr [_a]
/* 00000007 0007  0F BF 15 00 00 00 00 */	movsx edx, word ptr [_c]
/* 0000000E 000E  33 C9 */	xor ecx, ecx
/* 00000010 0010  A3 00 00 00 00 */	mov dword ptr [_ar], eax
/* 00000015 0015  8A 0D 00 00 00 00 */	mov cl, byte ptr [_b]
/* 0000001B 001B  33 C0 */	xor eax, eax
/* 0000001D 001D  66 A1 00 00 00 00 */	mov ax, word ptr [_d]
/* 00000023 0023  89 0D 04 00 00 00 */	mov dword ptr [_ar + 0x4], ecx
/* 00000029 0029  8B 0D 00 00 00 00 */	mov ecx, dword ptr [_e]
/* 0000002F 002F  89 15 08 00 00 00 */	mov dword ptr [_ar + 0x8], edx
/* 00000035 0035  8B 15 00 00 00 00 */	mov edx, dword ptr [_f]
/* 0000003B 003B  A3 0C 00 00 00 */	mov dword ptr [_ar + 0xc], eax
/* 00000040 0040  89 0D 10 00 00 00 */	mov dword ptr [_ar + 0x10], ecx
/* 00000046 0046  89 15 14 00 00 00 */	mov dword ptr [_ar + 0x14], edx
/* 0000004C 004C  C3 */	ret
/* 0000004D 004D  90 */	nop
/* 0000004E 004E  90 */	nop
/* 0000004F 004F  90 */	nop

