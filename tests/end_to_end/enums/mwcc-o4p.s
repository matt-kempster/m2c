.include "macros.inc"

.section .text  # 0x0 - 0x4c

.global test
test:
/* 00000000 00000000  2C 03 00 02 */	cmpwi r3, 2
/* 00000004 00000004  3C 80 00 00 */	lis r4, array@ha
/* 00000008 00000008  38 84 00 00 */	addi r4, r4, array@l
/* 0000000C 0000000C  41 82 00 28 */	beq .L00000034
/* 00000010 00000010  40 80 00 10 */	bge .L00000020
/* 00000014 00000014  2C 03 00 00 */	cmpwi r3, 0
/* 00000018 00000018  41 82 00 14 */	beq .L0000002C
/* 0000001C 0000001C  48 00 00 28 */	b .L00000044
.L00000020:
/* 00000020 00000020  2C 03 00 04 */	cmpwi r3, 4
/* 00000024 00000024  41 82 00 18 */	beq .L0000003C
/* 00000028 00000028  48 00 00 1C */	b .L00000044
.L0000002C:
/* 0000002C 0000002C  80 64 00 00 */	lwz r3, 0(r4)
/* 00000030 00000030  4E 80 00 20 */	blr 
.L00000034:
/* 00000034 00000034  80 64 00 04 */	lwz r3, 4(r4)
/* 00000038 00000038  4E 80 00 20 */	blr 
.L0000003C:
/* 0000003C 0000003C  80 64 00 08 */	lwz r3, 8(r4)
/* 00000040 00000040  4E 80 00 20 */	blr 
.L00000044:
/* 00000044 00000044  38 60 00 00 */	li r3, 0
/* 00000048 00000048  4E 80 00 20 */	blr 

.section .rodata  # 0x0 - 0x10

.global array
array:
	.word 0x00000004
	.word 0x00000003
	.word 0x00000002
	.word 0x00000000

