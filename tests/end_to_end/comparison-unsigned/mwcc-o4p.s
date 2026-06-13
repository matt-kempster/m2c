.include "macros.inc"

.section .text  # 0x0 - 0x30

.global test
test:
/* 00000000 00000000  7C 85 1A 78 */	xor r5, r4, r3
/* 00000004 00000004  7C 03 20 50 */	subf r0, r3, r4
/* 00000008 00000008  7C A5 00 34 */	cntlzw r5, r5
/* 0000000C 0000000C  7C 83 1B 38 */	orc r3, r4, r3
/* 00000010 00000010  7C 84 28 30 */	slw r4, r4, r5
/* 00000014 00000014  54 00 F8 7E */	srwi r0, r0, 1
/* 00000018 00000018  54 84 0F FE */	srwi r4, r4, 0x1f
/* 0000001C 0000001C  90 80 00 00 */	stw r4, global@sda21(r2)
/* 00000020 00000020  7C 00 18 50 */	subf r0, r0, r3
/* 00000024 00000024  54 00 0F FE */	srwi r0, r0, 0x1f
/* 00000028 00000028  90 00 00 00 */	stw r0, global@sda21(r2)
/* 0000002C 0000002C  4E 80 00 20 */	blr 

.section .sbss  # 0x0 - 0x4

.global global
global:
	.word 0x00000000

