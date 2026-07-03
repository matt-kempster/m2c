.section .text
test:
/* 00000000 0000  83 EC 08 */	sub esp, 8
/* 00000003 0003  D9 44 24 0C */	fld dword ptr [esp + 0xc]
/* 00000007 0007  E8 00 00 00 00 */	call __ftol
/* 0000000C 000C  DB 44 24 10 */	fild dword ptr [esp + 0x10]
/* 00000010 0010  A3 00 00 00 00 */	mov dword ptr [_globali], eax
/* 00000015 0015  8B 44 24 18 */	mov eax, dword ptr [esp + 0x18]
/* 00000019 0019  83 C0 03 */	add eax, 3
/* 0000001C 001C  C7 44 24 04 00 00 00 00 */	mov dword ptr [esp + 4], 0
/* 00000024 0024  D9 1D 00 00 00 00 */	fstp dword ptr [_globalf]
/* 0000002A 002A  D9 44 24 14 */	fld dword ptr [esp + 0x14]
/* 0000002E 002E  DC 05 00 00 00 00 */	fadd qword ptr [_real_4014000000000000]
/* 00000034 0034  89 44 24 00 */	mov dword ptr [esp], eax
/* 00000038 0038  DC 05 00 00 00 00 */	fadd qword ptr [_real_4015333333333333]
/* 0000003E 003E  DA 44 24 00 */	fiadd dword ptr [esp]
/* 00000042 0042  83 C4 08 */	add esp, 8
/* 00000045 0045  C3 */	ret
/* 00000046 0046  90 */	nop
/* 00000047 0047  90 */	nop
/* 00000048 0048  90 */	nop
/* 00000049 0049  90 */	nop
/* 0000004A 004A  90 */	nop
/* 0000004B 004B  90 */	nop
/* 0000004C 004C  90 */	nop
/* 0000004D 004D  90 */	nop
/* 0000004E 004E  90 */	nop
/* 0000004F 004F  90 */	nop

.section .rdata
_real_4015333333333333:
	.double 5.3

_real_4014000000000000:
	.double 5.0

