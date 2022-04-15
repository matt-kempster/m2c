.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches


glabel test
/* 000090 00400090 27BDFFF8 */  addiu $sp, $sp, -8
/* 000094 00400094 00802825 */  move  $a1, $a0
/* 000098 00400098 10A00009 */  beqz  $a1, .L004000C0
/* 00009C 0040009C 00000000 */   nop
/* 0000A0 004000A0 24010002 */  addiu $at, $zero, 2
/* 0000A4 004000A4 10A1000A */  beq   $a1, $at, .L004000D0
/* 0000A8 004000A8 00000000 */   nop
/* 0000AC 004000AC 24010004 */  addiu $at, $zero, 4
/* 0000B0 004000B0 10A1000B */  beq   $a1, $at, .L004000E0
/* 0000B4 004000B4 00000000 */   nop
/* 0000B8 004000B8 1000000D */  b     .L004000F0
/* 0000BC 004000BC 00000000 */   nop
.L004000C0:
/* 0000C0 004000C0 3C0E0040 */  lui   $t6, %hi(array)
/* 0000C4 004000C4 25CE0110 */  addiu $t6, $t6, %lo(array)
/* 0000C8 004000C8 1000000D */  b     .L00400100
/* 0000CC 004000CC 8DC20000 */   lw    $v0, ($t6)
.L004000D0:
/* 0000D0 004000D0 3C0F0040 */  lui   $t7, %hi(array)
/* 0000D4 004000D4 25EF0110 */  addiu $t7, $t7, %lo(array)
/* 0000D8 004000D8 10000009 */  b     .L00400100
/* 0000DC 004000DC 8DE20004 */   lw    $v0, 4($t7)
.L004000E0:
/* 0000E0 004000E0 3C180040 */  lui   $t8, %hi(array)
/* 0000E4 004000E4 27180110 */  addiu $t8, $t8, %lo(array)
/* 0000E8 004000E8 10000005 */  b     .L00400100
/* 0000EC 004000EC 8F020008 */   lw    $v0, 8($t8)
.L004000F0:
/* 0000F0 004000F0 10000003 */  b     .L00400100
/* 0000F4 004000F4 00001025 */   move  $v0, $zero
/* 0000F8 004000F8 10000001 */  b     .L00400100
/* 0000FC 004000FC 00000000 */   nop
.L00400100:
/* 000100 00400100 03E00008 */  jr    $ra
/* 000104 00400104 27BD0008 */   addiu $sp, $sp, 8

/* 000108 00400108 00000000 */  nop
/* 00010C 0040010C 00000000 */  nop

.section .rodata
.global array
array:
	.word 0x00000004
	.word 0x00000003
	.word 0x00000002
	.word 0x00000000

