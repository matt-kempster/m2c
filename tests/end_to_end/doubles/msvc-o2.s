.section .text
test:
/* 00000000 0000  DB 44 24 0C */	fild dword ptr [esp + 0xc]
/* 00000004 0004  DC 4C 24 04 */	fmul qword ptr [esp + 4]
/* 00000008 0008  DD 44 24 04 */	fld qword ptr [esp + 4]
/* 0000000C 000C  DC 74 24 10 */	fdiv qword ptr [esp + 0x10]
/* 00000010 0010  DE C1 */	faddp st(1)
/* 00000012 0012  DC 25 00 00 00 00 */	fsub qword ptr [_real_401c000000000000]
/* 00000018 0018  DC 54 24 10 */	fcom qword ptr [esp + 0x10]
/* 0000001C 001C  DF E0 */	fnstsw ax
/* 0000001E 001E  F6 C4 01 */	test ah, 1
/* 00000021 0021  75 25 */	jne .L00000048
/* 00000023 0023  DC 54 24 10 */	fcom qword ptr [esp + 0x10]
/* 00000027 0027  DF E0 */	fnstsw ax
/* 00000029 0029  F6 C4 40 */	test ah, 0x40
/* 0000002C 002C  75 1A */	jne .L00000048
/* 0000002E 002E  DC 1D 00 00 00 00 */	fcomp qword ptr [_real_4022000000000000]
/* 00000034 0034  DF E0 */	fnstsw ax
/* 00000036 0036  F6 C4 41 */	test ah, 0x41
/* 00000039 0039  74 0F */	je .L0000004A
/* 0000003B 003B  DD 05 00 00 00 00 */	fld qword ptr [_real_4018000000000000]
/* 00000041 0041  DD 15 00 00 00 00 */	fst qword ptr [_global]
/* 00000047 0047  C3 */	ret
.L00000048:
/* 00000048 0048  DD D8 */	fstp st(0)
.L0000004A:
/* 0000004A 004A  DD 05 00 00 00 00 */	fld qword ptr [_real_4014000000000000]
/* 00000050 0050  DD 15 00 00 00 00 */	fst qword ptr [_global]
/* 00000056 0056  C3 */	ret
/* 00000057 0057  90 */	nop
/* 00000058 0058  90 */	nop
/* 00000059 0059  90 */	nop
/* 0000005A 005A  90 */	nop
/* 0000005B 005B  90 */	nop
/* 0000005C 005C  90 */	nop
/* 0000005D 005D  90 */	nop
/* 0000005E 005E  90 */	nop
/* 0000005F 005F  90 */	nop

.section .rdata
_real_4014000000000000:
	.double 5.0

_real_4018000000000000:
	.double 6.0

_real_4022000000000000:
	.double 9.0

_real_401c000000000000:
	.double 7.0
