.include "macros.inc"

.section .text  # 0x0 - 0x5c

.global test
test:
/* 00000000 00000000  54 86 10 3A */	slwi r6, r4, 2
/* 00000004 00000004  7C A3 32 14 */	add r5, r3, r6
/* 00000008 00000008  80 A5 00 04 */	lwz r5, 4(r5)
/* 0000000C 0000000C  54 87 18 38 */	slwi r7, r4, 3
/* 00000010 00000010  38 06 00 04 */	addi r0, r6, 4
/* 00000014 00000014  90 A0 00 00 */	stw r5, glob@l(r0)
/* 00000018 00000018  7C 03 02 14 */	add r0, r3, r0
/* 0000001C 0000001C  38 A7 00 30 */	addi r5, r7, 0x30
/* 00000020 00000020  90 00 00 00 */	stw r0, glob@l(r0)
/* 00000024 00000024  7C C3 3A 14 */	add r6, r3, r7
/* 00000028 00000028  54 80 38 30 */	slwi r0, r4, 7
/* 0000002C 0000002C  80 C6 00 30 */	lwz r6, 0x30(r6)
/* 00000030 00000030  7C A3 2A 14 */	add r5, r3, r5
/* 00000034 00000034  7C 83 02 14 */	add r4, r3, r0
/* 00000038 00000038  90 C0 00 00 */	stw r6, glob@l(r0)
/* 0000003C 0000003C  38 03 00 48 */	addi r0, r3, 0x48
/* 00000040 00000040  90 A0 00 00 */	stw r5, glob@l(r0)
/* 00000044 00000044  80 84 00 7C */	lwz r4, 0x7c(r4)
/* 00000048 00000048  90 80 00 00 */	stw r4, glob@l(r0)
/* 0000004C 0000004C  80 63 00 48 */	lwz r3, 0x48(r3)
/* 00000050 00000050  90 60 00 00 */	stw r3, glob@l(r0)
/* 00000054 00000054  90 00 00 00 */	stw r0, glob@l(r0)
/* 00000058 00000058  4E 80 00 20 */	blr 

.section .sbss  # 0x0 - 0x4

.global glob
glob:
	.word 0x00000000
