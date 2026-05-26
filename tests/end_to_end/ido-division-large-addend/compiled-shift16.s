.set noat
.set noreorder

glabel test
/* 000000 3C01FFC0 */  lui   $at, 0xffc0
/* 000004 00811021 */  addu  $v0, $a0, $at
/* 000008 04410004 */  bgez  $v0, .L001C
/* 00000C 00027403 */   sra   $t6, $v0, 0x10
/* 000010 3401FFFF */  li    $at, 0xffff
/* 000014 00220821 */  addu  $at, $at, $v0
/* 000018 00017403 */  sra   $t6, $at, 0x10
.L001C:
/* 00001C 03E00008 */  jr    $ra
/* 000020 01C01025 */   move  $v0, $t6
