.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches


glabel foo
/* 0000A0 004000A0 AFA50004 */  sw    $a1, 4($sp)
/* 0000A4 004000A4 03E00008 */  jr    $ra
/* 0000A8 004000A8 00801025 */   move  $v0, $a0

glabel test
/* 0000AC 004000AC 27BDFFE0 */  addiu $sp, $sp, -0x20
/* 0000B0 004000B0 AFB10018 */  sw    $s1, 0x18($sp)
/* 0000B4 004000B4 AFB00014 */  sw    $s0, 0x14($sp)
/* 0000B8 004000B8 00A08825 */  move  $s1, $a1
/* 0000BC 004000BC AFBF001C */  sw    $ra, 0x1c($sp)
/* 0000C0 004000C0 AFA40020 */  sw    $a0, 0x20($sp)
/* 0000C4 004000C4 AFA60028 */  sw    $a2, 0x28($sp)
/* 0000C8 004000C8 00008025 */  move  $s0, $zero
.L004000CC:
/* 0000CC 004000CC 02002025 */  move  $a0, $s0
/* 0000D0 004000D0 0C100028 */  jal   foo
/* 0000D4 004000D4 02202825 */   move  $a1, $s1
/* 0000D8 004000D8 1000FFFC */  b     .L004000CC
/* 0000DC 004000DC 00408025 */   move  $s0, $v0
/* 0000E0 004000E0 8FBF001C */  lw    $ra, 0x1c($sp)
/* 0000E4 004000E4 8FB00014 */  lw    $s0, 0x14($sp)
/* 0000E8 004000E8 8FB10018 */  lw    $s1, 0x18($sp)
/* 0000EC 004000EC 03E00008 */  jr    $ra
/* 0000F0 004000F0 27BD0020 */   addiu $sp, $sp, 0x20

/* 0000F4 004000F4 00000000 */  nop
/* 0000F8 004000F8 00000000 */  nop
/* 0000FC 004000FC 00000000 */  nop
