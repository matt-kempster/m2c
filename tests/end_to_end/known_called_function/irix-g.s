.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches


glabel foo
/* 000090 00400090 AFA50004 */  sw    $a1, 4($sp)
/* 000094 00400094 10000003 */  b     .L004000A4
/* 000098 00400098 00801025 */   move  $v0, $a0
/* 00009C 0040009C 10000001 */  b     .L004000A4
/* 0000A0 004000A0 00000000 */   nop
.L004000A4:
/* 0000A4 004000A4 03E00008 */  jr    $ra
/* 0000A8 004000A8 00000000 */   nop

glabel test
/* 0000AC 004000AC 27BDFFE0 */  addiu $sp, $sp, -0x20
/* 0000B0 004000B0 AFBF0014 */  sw    $ra, 0x14($sp)
/* 0000B4 004000B4 AFA40020 */  sw    $a0, 0x20($sp)
/* 0000B8 004000B8 AFA50024 */  sw    $a1, 0x24($sp)
/* 0000BC 004000BC AFA60028 */  sw    $a2, 0x28($sp)
/* 0000C0 004000C0 AFA0001C */  sw    $zero, 0x1c($sp)
.L004000C4:
/* 0000C4 004000C4 8FA4001C */  lw    $a0, 0x1c($sp)
/* 0000C8 004000C8 0C100024 */  jal   foo
/* 0000CC 004000CC 8FA50024 */   lw    $a1, 0x24($sp)
/* 0000D0 004000D0 AFA2001C */  sw    $v0, 0x1c($sp)
/* 0000D4 004000D4 1000FFFB */  b     .L004000C4
/* 0000D8 004000D8 00000000 */   nop
/* 0000DC 004000DC 10000001 */  b     .L004000E4
/* 0000E0 004000E0 00000000 */   nop
.L004000E4:
/* 0000E4 004000E4 8FBF0014 */  lw    $ra, 0x14($sp)
/* 0000E8 004000E8 27BD0020 */  addiu $sp, $sp, 0x20
/* 0000EC 004000EC 03E00008 */  jr    $ra
/* 0000F0 004000F0 00000000 */   nop

/* 0000F4 004000F4 00000000 */  nop
/* 0000F8 004000F8 00000000 */  nop
/* 0000FC 004000FC 00000000 */  nop
