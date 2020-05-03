.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches


glabel test
/* 0000B0 004000B0 00057080 */  sll   $t6, $a1, 2
/* 0000B4 004000B4 008E1021 */  addu  $v0, $a0, $t6
/* 0000B8 004000B8 8C4F0004 */  lw    $t7, 4($v0)
/* 0000BC 004000BC 3C060041 */  lui   $a2, %hi(D_410100)
/* 0000C0 004000C0 24C60100 */  addiu $a2, $a2, %lo(D_410100)
/* 0000C4 004000C4 24580004 */  addiu $t8, $v0, 4
/* 0000C8 004000C8 0005C8C0 */  sll   $t9, $a1, 3
/* 0000CC 004000CC ACCF0000 */  sw    $t7, ($a2)
/* 0000D0 004000D0 ACD80000 */  sw    $t8, ($a2)
/* 0000D4 004000D4 00991821 */  addu  $v1, $a0, $t9
/* 0000D8 004000D8 8C680030 */  lw    $t0, 0x30($v1)
/* 0000DC 004000DC 24690030 */  addiu $t1, $v1, 0x30
/* 0000E0 004000E0 000551C0 */  sll   $t2, $a1, 7
/* 0000E4 004000E4 ACC80000 */  sw    $t0, ($a2)
/* 0000E8 004000E8 ACC90000 */  sw    $t1, ($a2)
/* 0000EC 004000EC 008A5821 */  addu  $t3, $a0, $t2
/* 0000F0 004000F0 8D6C007C */  lw    $t4, 0x7c($t3)
/* 0000F4 004000F4 03E00008 */  jr    $ra
/* 0000F8 004000F8 ACCC0000 */   sw    $t4, ($a2)

/* 0000FC 004000FC 00000000 */  nop
