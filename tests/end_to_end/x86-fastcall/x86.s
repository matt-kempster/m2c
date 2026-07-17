# Goal: infer the ECX/EDX fastcall arguments before either register is written.
# Generated from orig.c with MSVC6 /O1 and msvc_disasm.
.section .text
# MSVC symbol: "@test@12"
test:
/* 00000000 0000  8D 04 11 */	lea eax, [ecx + edx]
/* 00000003 0003  03 44 24 04 */	add eax, dword ptr [esp + 4]
/* 00000007 0007  C2 04 00 */	ret 4
