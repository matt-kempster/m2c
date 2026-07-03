.section .rdata
_D_400120:
	.long 0x41200000
	.long 0x41300000
	.long 0x41400000
_D_40012C:
	.long 0x41600000
	.long 0x41700000
	.long 0x41800000
	.long 0x41880000
	.long 0x41900000

.section .data
_D_410150:
	.long 0x3F9D70A4
_D_410154:
	.long 0x40400000
	.long 0x40800000
	.long 0x40A00000
_D_410160:
	.long 0x40C00000
	.long 0x40E00000
	.long 0x41000000

.section .text
test:
/* 0000003C 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000040 0004  D9 04 85 00 00 00 00 */	fld dword ptr [eax*4 + _D_400120]
/* 00000047 000B  D8 04 85 00 00 00 00 */	fadd dword ptr [eax*4 + _D_410160]
/* 0000004E 0012  D9 14 85 00 00 00 00 */	fst dword ptr [eax*4 + _D_410170]
/* 00000055 0019  D9 C0 */	fld st(0)
/* 00000057 001B  D8 0D 00 00 00 00 */	fmul dword ptr [_real_40b570a4]
/* 0000005D 0021  D8 04 85 00 00 00 00 */	fadd dword ptr [eax*4 + _D_40012C]
/* 00000064 0028  D8 0D 00 00 00 00 */	fmul dword ptr [_D_410150]
/* 0000006A 002E  D9 15 00 00 00 00 */	fst dword ptr [_D_410150]
/* 00000070 0034  DE F1 */	fdivrp st(1)
/* 00000072 0036  C3 */	ret
/* 00000073 0037  90 */	nop
/* 00000074 0038  90 */	nop
/* 00000075 0039  90 */	nop
/* 00000076 003A  90 */	nop
/* 00000077 003B  90 */	nop
/* 00000078 003C  90 */	nop
/* 00000079 003D  90 */	nop
/* 0000007A 003E  90 */	nop
/* 0000007B 003F  90 */	nop

.section .rdata
_real_40b570a4:
	.float 5.670000076293945

.section .bss
_D_410170:
	.space 0xC

