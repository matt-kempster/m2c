.set noat
.set noreorder

glabel test
/* 000000 3C01FFC0 */  lui   $at, 0xffc0
/* 000004 00811021 */  addu  $v0, $a0, $at
/* 000008 04410003 */  bgez  $v0, .L0018
/* 00000C 00027143 */   sra   $t6, $v0, 5
/* 000010 2441001F */  addiu $at, $v0, 0x1f
/* 000014 00017143 */  sra   $t6, $at, 5
.L0018:
/* 000018 03E00008 */  jr    $ra
/* 00001C 00201025 */   move  $v0, $at
