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
/* 0000002E 002E  D8 05 00 00 00 00 */	fadd dword ptr [_real_40a00000]
/* 00000034 0034  89 44 24 00 */	mov dword ptr [esp], eax
/* 00000038 0038  DA 44 24 00 */	fiadd dword ptr [esp]
/* 0000003C 003C  83 C4 08 */	add esp, 8
/* 0000003F 003F  C3 */	ret

.section .rdata
_real_40a00000:
	.float 5.0

