.section .text
test:
/* 00000000 0000  DD 44 24 04 */	fld qword ptr [esp + 4]
/* 00000004 0004  DD 44 24 0C */	fld qword ptr [esp + 0xc]
/* 00000008 0008  E8 00 00 00 00 */	call __CIpow
/* 0000000D 000D  DC C0 */	fadd st(0), st(0)
/* 0000000F 000F  DD 44 24 04 */	fld qword ptr [esp + 4]
/* 00000013 0013  DD 44 24 0C */	fld qword ptr [esp + 0xc]
/* 00000017 0017  E8 00 00 00 00 */	call __CIfmod
/* 0000001C 001C  DE C1 */	faddp st(1)
/* 0000001E 001E  C3 */	ret
