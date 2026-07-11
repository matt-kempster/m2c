.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000008 0008  56 */	push esi
/* 00000009 0009  8B 74 81 04 */	mov esi, dword ptr [ecx + eax*4 + 4]
/* 0000000D 000D  8D 54 81 04 */	lea edx, [ecx + eax*4 + 4]
/* 00000011 0011  89 35 00 00 00 00 */	mov dword ptr [_glob], esi
/* 00000017 0017  89 15 00 00 00 00 */	mov dword ptr [_glob], edx
/* 0000001D 001D  8B 74 C1 30 */	mov esi, dword ptr [ecx + eax*8 + 0x30]
/* 00000021 0021  8D 54 C1 30 */	lea edx, [ecx + eax*8 + 0x30]
/* 00000025 0025  89 35 00 00 00 00 */	mov dword ptr [_glob], esi
/* 0000002B 002B  89 15 00 00 00 00 */	mov dword ptr [_glob], edx
/* 00000031 0031  C1 E0 07 */	shl eax, 7
/* 00000034 0034  8B 44 08 7C */	mov eax, dword ptr [eax + ecx + 0x7c]
/* 00000038 0038  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 0000003D 003D  8D 41 48 */	lea eax, [ecx + 0x48]
/* 00000040 0040  8B 49 48 */	mov ecx, dword ptr [ecx + 0x48]
/* 00000043 0043  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 00000049 0049  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 0000004E 004E  5E */	pop esi
/* 0000004F 004F  C3 */	ret
