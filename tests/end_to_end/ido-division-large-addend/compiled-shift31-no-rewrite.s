.set noat
.set noreorder

glabel test
/* 000000 3C01FFC0 */  lui   $at, 0xffc0
/* 000004 00811021 */  addu  $v0, $a0, $at
/* 000008 04410005 */  bgez  $v0, .L0020
/* 00000C 000277C3 */   sra   $t6, $v0, 0x1f
/* 000010 3C018000 */  lui   $at, 0x8000
/* 000014 2421FFFF */  addiu $at, $at, -1
/* 000018 00220821 */  addu  $at, $at, $v0
/* 00001C 000177C3 */  sra   $t6, $at, 0x1f
.L0020:
/* 000020 03E00008 */  jr    $ra
/* 000024 01C01025 */   move  $v0, $t6
