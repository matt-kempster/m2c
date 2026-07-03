.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  A3 00 00 00 00 */	mov dword ptr [_y], eax
/* 00000009 0009  8D 0C 00 */	lea ecx, [eax + eax]
/* 0000000C 000C  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 00000012 0012  8D 14 40 */	lea edx, [eax + eax*2]
/* 00000015 0015  89 15 00 00 00 00 */	mov dword ptr [_y], edx
/* 0000001B 001B  8D 0C 85 00 00 00 00 */	lea ecx, [eax*4]
/* 00000022 0022  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 00000028 0028  8D 14 80 */	lea edx, [eax + eax*4]
/* 0000002B 002B  89 15 00 00 00 00 */	mov dword ptr [_y], edx
/* 00000031 0031  8D 0C 40 */	lea ecx, [eax + eax*2]
/* 00000034 0034  D1 E1 */	shl ecx, 1
/* 00000036 0036  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 0000003C 003C  8D 14 C5 00 00 00 00 */	lea edx, [eax*8]
/* 00000043 0043  2B D0 */	sub edx, eax
/* 00000045 0045  89 15 00 00 00 00 */	mov dword ptr [_y], edx
/* 0000004B 004B  8D 0C C5 00 00 00 00 */	lea ecx, [eax*8]
/* 00000052 0052  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 00000058 0058  8D 14 C0 */	lea edx, [eax + eax*8]
/* 0000005B 005B  89 15 00 00 00 00 */	mov dword ptr [_y], edx
/* 00000061 0061  8D 0C 80 */	lea ecx, [eax + eax*4]
/* 00000064 0064  D1 E1 */	shl ecx, 1
/* 00000066 0066  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 0000006C 006C  8D 14 80 */	lea edx, [eax + eax*4]
/* 0000006F 006F  8D 0C 50 */	lea ecx, [eax + edx*2]
/* 00000072 0072  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 00000078 0078  8D 14 40 */	lea edx, [eax + eax*2]
/* 0000007B 007B  C1 E2 02 */	shl edx, 2
/* 0000007E 007E  89 15 00 00 00 00 */	mov dword ptr [_y], edx
/* 00000084 0084  8D 0C 40 */	lea ecx, [eax + eax*2]
/* 00000087 0087  8D 14 88 */	lea edx, [eax + ecx*4]
/* 0000008A 008A  89 15 00 00 00 00 */	mov dword ptr [_y], edx
/* 00000090 0090  8D 0C C5 00 00 00 00 */	lea ecx, [eax*8]
/* 00000097 0097  2B C8 */	sub ecx, eax
/* 00000099 0099  D1 E1 */	shl ecx, 1
/* 0000009B 009B  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 000000A1 00A1  8D 0C 40 */	lea ecx, [eax + eax*2]
/* 000000A4 00A4  8D 14 89 */	lea edx, [ecx + ecx*4]
/* 000000A7 00A7  89 15 00 00 00 00 */	mov dword ptr [_y], edx
/* 000000AD 00AD  8B C8 */	mov ecx, eax
/* 000000AF 00AF  C1 E1 04 */	shl ecx, 4
/* 000000B2 00B2  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 000000B8 00B8  8B D0 */	mov edx, eax
/* 000000BA 00BA  C1 E2 04 */	shl edx, 4
/* 000000BD 00BD  03 D0 */	add edx, eax
/* 000000BF 00BF  89 15 00 00 00 00 */	mov dword ptr [_y], edx
/* 000000C5 00C5  8D 0C C0 */	lea ecx, [eax + eax*8]
/* 000000C8 00C8  D1 E1 */	shl ecx, 1
/* 000000CA 00CA  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 000000D0 00D0  8D 14 C0 */	lea edx, [eax + eax*8]
/* 000000D3 00D3  8D 0C 50 */	lea ecx, [eax + edx*2]
/* 000000D6 00D6  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 000000DC 00DC  8D 14 80 */	lea edx, [eax + eax*4]
/* 000000DF 00DF  C1 E2 02 */	shl edx, 2
/* 000000E2 00E2  89 15 00 00 00 00 */	mov dword ptr [_y], edx
/* 000000E8 00E8  8D 0C C5 00 00 00 00 */	lea ecx, [eax*8]
/* 000000EF 00EF  2B C8 */	sub ecx, eax
/* 000000F1 00F1  8D 0C 49 */	lea ecx, [ecx + ecx*2]
/* 000000F4 00F4  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 000000FA 00FA  8D 14 80 */	lea edx, [eax + eax*4]
/* 000000FD 00FD  8D 04 50 */	lea eax, [eax + edx*2]
/* 00000100 0100  D1 E0 */	shl eax, 1
/* 00000102 0102  A3 00 00 00 00 */	mov dword ptr [_y], eax
/* 00000107 0107  C3 */	ret
/* 00000108 0108  90 */	nop
/* 00000109 0109  90 */	nop
/* 0000010A 010A  90 */	nop
/* 0000010B 010B  90 */	nop
/* 0000010C 010C  90 */	nop
/* 0000010D 010D  90 */	nop
/* 0000010E 010E  90 */	nop
/* 0000010F 010F  90 */	nop

