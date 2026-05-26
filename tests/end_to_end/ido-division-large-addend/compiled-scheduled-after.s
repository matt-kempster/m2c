.set noat
.set noreorder

glabel test
/* 000180 8C830000 */  lw    $v1, ($a0)
/* 000184 3C01FFC0 */  lui   $at, 0xffc0
/* 000188 8C850004 */  lw    $a1, 4($a0)
/* 00018C 00617021 */  addu  $t6, $v1, $at
/* 000190 8C860008 */  lw    $a2, 8($a0)
/* 000194 05C10003 */  bgez  $t6, .L01A4
/* 000198 000E7943 */   sra   $t7, $t6, 5
/* 00019C 25C1001F */  addiu $at, $t6, 0x1f
/* 0001A0 00017943 */  sra   $t7, $at, 5
.L01A4:
/* 0001A4 00AFC023 */  subu  $t8, $a1, $t7
/* 0001A8 03E00008 */  jr    $ra
/* 0001AC 03061021 */   addu  $v0, $t8, $a2
