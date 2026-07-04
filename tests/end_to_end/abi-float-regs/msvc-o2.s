.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  33 C9 */	xor ecx, ecx
/* 00000006 0006  DD 05 00 00 00 00 */	fld qword ptr [_real_3ff0000000000000]
/* 0000000C 000C  83 EC 08 */	sub esp, 8
/* 0000000F 000F  3B C1 */	cmp eax, ecx
/* 00000011 0011  74 11 */	je .L00000024
.L00000013:
/* 00000013 0013  89 44 24 00 */	mov dword ptr [esp], eax
/* 00000017 0017  89 4C 24 04 */	mov dword ptr [esp + 4], ecx
/* 0000001B 001B  DF 6C 24 00 */	fild qword ptr [esp]
/* 0000001F 001F  48 */	dec eax
/* 00000020 0020  DE C9 */	fmulp st(1)
/* 00000022 0022  75 EF */	jne .L00000013
.L00000024:
/* 00000024 0024  83 C4 08 */	add esp, 8
/* 00000027 0027  C3 */	ret
/* 00000028 0028  90 */	nop
/* 00000029 0029  90 */	nop
/* 0000002A 002A  90 */	nop
/* 0000002B 002B  90 */	nop
/* 0000002C 002C  90 */	nop
/* 0000002D 002D  90 */	nop
/* 0000002E 002E  90 */	nop
/* 0000002F 002F  90 */	nop

.section .rdata
_real_3ff0000000000000:
	.double 1.0

