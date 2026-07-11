.section .text
test:
/* 00000000 0000  C7 05 00 00 00 00 9A 99 99 3F */	mov dword ptr [_a], 0x3f99999a
/* 0000000A 000A  C7 05 00 00 00 00 00 00 00 00 */	mov dword ptr [_b], 0
/* 00000014 0014  C7 05 04 00 00 00 00 00 2A 40 */	mov dword ptr [_b + 0x4], 0x402a0000
/* 0000001E 001E  C7 05 00 00 00 00 00 00 00 60 */	mov dword ptr [_c], 0x60000000
/* 00000028 0028  C7 05 04 00 00 00 B8 13 0A 42 */	mov dword ptr [_c + 0x4], 0x420a13b8
/* 00000032 0032  C7 05 00 00 00 00 00 00 00 00 */	mov dword ptr [_d], "??_C@_09DAMD@?$CChello?$CC?6?$AB?$AA@"
/* 0000003C 003C  C3 */	ret
/* 0000003D 003D  90 */	nop
/* 0000003E 003E  90 */	nop
/* 0000003F 003F  90 */	nop

.section .data
"??_C@_09DAMD@?$CChello?$CC?6?$AB?$AA@":
	.long 0x6C656822
	.long 0x0A226F6C
	.byte 0x01
	.byte 0x00
