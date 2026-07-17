# Goal: infer callee cleanup structurally from the stdcall `ret 8`.
# Generated from orig.c with MSVC6 /O1 and msvc_disasm.
.section .text
# MSVC symbol: "_test@8"
test:
/* 00000000 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000004 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000008 0008  03 C1 */	add eax, ecx
/* 0000000A 000A  C2 08 00 */	ret 8
