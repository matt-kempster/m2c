# Goal: decode a no-base scaled-index store as a generic address expression.
# Generated from orig.c with MSVC6 /O1 and msvc_disasm.
.section .text
test:
/* 00000000 0000  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000004 0004  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000008 0008  89 04 8D 00 00 00 00 */	mov dword ptr [ecx*4 + _table], eax
/* 0000000F 000F  C6 44 41 01 07 */	mov byte ptr [ecx + eax*2 + 1], 7
/* 00000014 0014  C3 */	ret
