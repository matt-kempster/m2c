.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches


glabel func_00400090
/* 000090 00400090 03E00008 */  jr    $ra
/* 000094 00400094 24820001 */   addiu $v0, $a0, 1

glabel test
/* 000098 00400098 27BDFFD8 */  addiu $sp, $sp, -0x28
/* 00009C 0040009C 00857021 */  addu  $t6, $a0, $a1
/* 0000A0 004000A0 AFBF0014 */  sw    $ra, 0x14($sp)
/* 0000A4 004000A4 15C0000B */  bnez  $t6, .L004000D4
/* 0000A8 004000A8 AFA70034 */   sw    $a3, 0x34($sp)
/* 0000AC 004000AC 00A62021 */  addu  $a0, $a1, $a2
/* 0000B0 004000B0 54800009 */  bnezl $a0, .L004000D8
/* 0000B4 004000B4 24030001 */   addiu $v1, $zero, 1
/* 0000B8 004000B8 0C100024 */  jal   func_00400090
/* 0000BC 004000BC AFA0001C */   sw    $zero, 0x1c($sp)
/* 0000C0 004000C0 14400004 */  bnez  $v0, .L004000D4
/* 0000C4 004000C4 8FA3001C */   lw    $v1, 0x1c($sp)
/* 0000C8 004000C8 8FAF0034 */  lw    $t7, 0x34($sp)
/* 0000CC 004000CC 51E00003 */  beql  $t7, $zero, .L004000DC
/* 0000D0 004000D0 8FBF0014 */   lw    $ra, 0x14($sp)
.L004000D4:
/* 0000D4 004000D4 24030001 */  addiu $v1, $zero, 1
.L004000D8:
/* 0000D8 004000D8 8FBF0014 */  lw    $ra, 0x14($sp)
.L004000DC:
/* 0000DC 004000DC 27BD0028 */  addiu $sp, $sp, 0x28
/* 0000E0 004000E0 00601025 */  move  $v0, $v1
/* 0000E4 004000E4 03E00008 */  jr    $ra
/* 0000E8 004000E8 00000000 */   nop

/* 0000EC 004000EC 00000000 */  nop
