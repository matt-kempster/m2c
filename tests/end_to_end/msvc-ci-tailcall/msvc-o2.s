.section .text
test:
/* 00000000 0000  DD 44 24 04 */	fld qword ptr [esp + 4]
/* 00000004 0004  DD 44 24 0C */	fld qword ptr [esp + 0xc]
/* 00000008 0008  E9 00 00 00 00 */	jmp __CIpow
