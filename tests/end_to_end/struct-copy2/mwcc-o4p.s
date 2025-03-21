.include "macros.inc"

.section .text  # 0x0 - 0x13c

.global test_0
test_0:
/* 00000000 00000000  80 A4 00 00 */	lwz r5, 0(r4)
/* 00000004 00000004  80 04 00 04 */	lwz r0, 4(r4)
/* 00000008 00000008  90 A3 00 00 */	stw r5, 0(r3)
/* 0000000C 0000000C  90 03 00 04 */	stw r0, 4(r3)
/* 00000010 00000010  4E 80 00 20 */	blr 

.global test_1
test_1:
/* 00000014 00000014  80 A4 00 00 */	lwz r5, 0(r4)
/* 00000018 00000018  80 04 00 04 */	lwz r0, 4(r4)
/* 0000001C 0000001C  90 A3 00 00 */	stw r5, 0(r3)
/* 00000020 00000020  90 03 00 04 */	stw r0, 4(r3)
/* 00000024 00000024  88 04 00 08 */	lbz r0, 8(r4)
/* 00000028 00000028  98 03 00 08 */	stb r0, 8(r3)
/* 0000002C 0000002C  4E 80 00 20 */	blr 

.global test_2
test_2:
/* 00000030 00000030  80 A4 00 00 */	lwz r5, 0(r4)
/* 00000034 00000034  80 04 00 04 */	lwz r0, 4(r4)
/* 00000038 00000038  90 A3 00 00 */	stw r5, 0(r3)
/* 0000003C 0000003C  90 03 00 04 */	stw r0, 4(r3)
/* 00000040 00000040  A0 04 00 08 */	lhz r0, 8(r4)
/* 00000044 00000044  B0 03 00 08 */	sth r0, 8(r3)
/* 00000048 00000048  4E 80 00 20 */	blr 

.global test_3
test_3:
/* 0000004C 0000004C  80 A4 00 00 */	lwz r5, 0(r4)
/* 00000050 00000050  80 04 00 04 */	lwz r0, 4(r4)
/* 00000054 00000054  90 A3 00 00 */	stw r5, 0(r3)
/* 00000058 00000058  90 03 00 04 */	stw r0, 4(r3)
/* 0000005C 0000005C  A0 04 00 08 */	lhz r0, 8(r4)
/* 00000060 00000060  B0 03 00 08 */	sth r0, 8(r3)
/* 00000064 00000064  88 04 00 0A */	lbz r0, 0xa(r4)
/* 00000068 00000068  98 03 00 0A */	stb r0, 0xa(r3)
/* 0000006C 0000006C  4E 80 00 20 */	blr 

.global test_4
test_4:
/* 00000070 00000070  80 A4 00 00 */	lwz r5, 0(r4)
/* 00000074 00000074  80 04 00 04 */	lwz r0, 4(r4)
/* 00000078 00000078  90 A3 00 00 */	stw r5, 0(r3)
/* 0000007C 0000007C  90 03 00 04 */	stw r0, 4(r3)
/* 00000080 00000080  80 04 00 08 */	lwz r0, 8(r4)
/* 00000084 00000084  90 03 00 08 */	stw r0, 8(r3)
/* 00000088 00000088  4E 80 00 20 */	blr 

.global test_5
test_5:
/* 0000008C 0000008C  80 A4 00 00 */	lwz r5, 0(r4)
/* 00000090 00000090  80 04 00 04 */	lwz r0, 4(r4)
/* 00000094 00000094  90 A3 00 00 */	stw r5, 0(r3)
/* 00000098 00000098  90 03 00 04 */	stw r0, 4(r3)
/* 0000009C 0000009C  80 04 00 08 */	lwz r0, 8(r4)
/* 000000A0 000000A0  90 03 00 08 */	stw r0, 8(r3)
/* 000000A4 000000A4  88 04 00 0C */	lbz r0, 0xc(r4)
/* 000000A8 000000A8  98 03 00 0C */	stb r0, 0xc(r3)
/* 000000AC 000000AC  4E 80 00 20 */	blr 

.global test_6
test_6:
/* 000000B0 000000B0  80 A4 00 00 */	lwz r5, 0(r4)
/* 000000B4 000000B4  80 04 00 04 */	lwz r0, 4(r4)
/* 000000B8 000000B8  90 A3 00 00 */	stw r5, 0(r3)
/* 000000BC 000000BC  90 03 00 04 */	stw r0, 4(r3)
/* 000000C0 000000C0  80 04 00 08 */	lwz r0, 8(r4)
/* 000000C4 000000C4  90 03 00 08 */	stw r0, 8(r3)
/* 000000C8 000000C8  A0 04 00 0C */	lhz r0, 0xc(r4)
/* 000000CC 000000CC  B0 03 00 0C */	sth r0, 0xc(r3)
/* 000000D0 000000D0  4E 80 00 20 */	blr 

.global test_7
test_7:
/* 000000D4 000000D4  81 04 00 00 */	lwz r8, 0(r4)
/* 000000D8 000000D8  3C A0 00 00 */	lis r5, s7@ha
/* 000000DC 000000DC  80 04 00 04 */	lwz r0, 4(r4)
/* 000000E0 000000E0  3C C0 00 00 */	lis r6, d7@ha
/* 000000E4 000000E4  38 E5 00 00 */	addi r7, r5, s7@l
/* 000000E8 000000E8  91 03 00 00 */	stw r8, 0(r3)
/* 000000EC 000000EC  39 06 00 00 */	addi r8, r6, d7@l
/* 000000F0 000000F0  90 03 00 04 */	stw r0, 4(r3)
/* 000000F4 000000F4  80 04 00 08 */	lwz r0, 8(r4)
/* 000000F8 000000F8  90 03 00 08 */	stw r0, 8(r3)
/* 000000FC 000000FC  A0 04 00 0C */	lhz r0, 0xc(r4)
/* 00000100 00000100  B0 03 00 0C */	sth r0, 0xc(r3)
/* 00000104 00000104  88 04 00 0E */	lbz r0, 0xe(r4)
/* 00000108 00000108  98 03 00 0E */	stb r0, 0xe(r3)
/* 0000010C 0000010C  80 C7 00 00 */	lwz r6, 0(r7)
/* 00000110 00000110  80 A7 00 04 */	lwz r5, 4(r7)
/* 00000114 00000114  80 87 00 08 */	lwz r4, 8(r7)
/* 00000118 00000118  A0 67 00 0C */	lhz r3, 0xc(r7)
/* 0000011C 0000011C  88 07 00 0E */	lbz r0, 0xe(r7)
/* 00000120 00000120  90 C8 00 00 */	stw r6, 0(r8)
/* 00000124 00000124  90 A8 00 04 */	stw r5, 4(r8)
/* 00000128 00000128  90 88 00 08 */	stw r4, 8(r8)
/* 0000012C 0000012C  B0 68 00 0C */	sth r3, 0xc(r8)
/* 00000130 00000130  98 08 00 0E */	stb r0, 0xe(r8)
/* 00000134 00000134  4E 80 00 20 */	blr 

.global test
test:
/* 00000138 00000138  4E 80 00 20 */	blr 

.section .bss  # 0x0 - 0x1f

.global d7
d7:
	.word 0x00000000
	.word 0x00000000
	.word 0x00000000
	.word 0x00000000

.global s7
s7:
	.word 0x00000000
	.word 0x00000000
	.word 0x00000000
	.bytes 0x0, 0x0, 0x0

