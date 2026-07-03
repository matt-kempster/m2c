.section .text
test:
/* 00000000 0000  83 EC 08 */	sub esp, 8
/* 00000003 0003  D9 05 00 00 00 00 */	fld dword ptr [_flt]
/* 00000009 0009  E8 00 00 00 00 */	call __ftol
/* 0000000E 000E  A3 00 00 00 00 */	mov dword ptr [_u], eax
/* 00000013 0013  DD 05 00 00 00 00 */	fld qword ptr [_dbl]
/* 00000019 0019  E8 00 00 00 00 */	call __ftol
/* 0000001E 001E  A3 00 00 00 00 */	mov dword ptr [_u], eax
/* 00000023 0023  A1 00 00 00 00 */	mov eax, dword ptr [_u]
/* 00000028 0028  89 44 24 00 */	mov dword ptr [esp], eax
/* 0000002C 002C  33 C0 */	xor eax, eax
/* 0000002E 002E  89 44 24 04 */	mov dword ptr [esp + 4], eax
/* 00000032 0032  DF 6C 24 00 */	fild qword ptr [esp]
/* 00000036 0036  DD 1D 00 00 00 00 */	fstp qword ptr [_dbl]
/* 0000003C 003C  8B 0D 00 00 00 00 */	mov ecx, dword ptr [_u]
/* 00000042 0042  89 4C 24 00 */	mov dword ptr [esp], ecx
/* 00000046 0046  89 44 24 04 */	mov dword ptr [esp + 4], eax
/* 0000004A 004A  DF 6C 24 00 */	fild qword ptr [esp]
/* 0000004E 004E  D9 1D 00 00 00 00 */	fstp dword ptr [_flt]
/* 00000054 0054  83 C4 08 */	add esp, 8
/* 00000057 0057  C3 */	ret
/* 00000058 0058  90 */	nop
/* 00000059 0059  90 */	nop
/* 0000005A 005A  90 */	nop
/* 0000005B 005B  90 */	nop
/* 0000005C 005C  90 */	nop
/* 0000005D 005D  90 */	nop
/* 0000005E 005E  90 */	nop
/* 0000005F 005F  90 */	nop

