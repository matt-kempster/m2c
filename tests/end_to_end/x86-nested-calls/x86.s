# Goal: keep an outer-call argument pending across the nested inner call.
# Generated from orig.c with MSVC6 /O2 and msvc_disasm.
.section .text
test:
/* 00000000 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000004 0004  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000008 0008  50 */	push eax
/* 00000009 0009  51 */	push ecx
/* 0000000A 000A  E8 00 00 00 00 */	call _inner
/* 0000000F 000F  83 C4 04 */	add esp, 4
/* 00000012 0012  50 */	push eax
/* 00000013 0013  E8 00 00 00 00 */	call _outer
/* 00000018 0018  83 C4 08 */	add esp, 8
/* 0000001B 001B  C3 */	ret
