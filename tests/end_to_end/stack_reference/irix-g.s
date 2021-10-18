.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches


glabel test
/* 000090 00400090 AFA40000 */  sw    $a0, ($sp)
/* 000094 00400094 AFA50004 */  sw    $a1, 4($sp)
/* 000098 00400098 27AE0004 */  addiu $t6, $sp, 4
/* 00009C 0040009C 27AF0000 */  addiu $t7, $sp, 0
/* 0000A0 004000A0 01CF1023 */  subu  $v0, $t6, $t7
/* 0000A4 004000A4 10000003 */  b     .L004000B4
/* 0000A8 004000A8 00021083 */   sra   $v0, $v0, 2
/* 0000AC 004000AC 10000001 */  b     .L004000B4
/* 0000B0 004000B0 00000000 */   nop
.L004000B4:
/* 0000B4 004000B4 03E00008 */  jr    $ra
/* 0000B8 004000B8 00000000 */   nop

/* 0000BC 004000BC 00000000 */  nop
