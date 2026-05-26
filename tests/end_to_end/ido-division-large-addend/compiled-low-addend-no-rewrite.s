.set noat
.set noreorder

glabel test
/* 000000 3C01FFC0 */  lui   $at, 0xffc0
/* 000004 3421001F */  ori   $at, $at, 0x1f
/* 000008 00811021 */  addu  $v0, $a0, $at
/* 00000C 04410003 */  bgez  $v0, .L001C
/* 000010 00027143 */   sra   $t6, $v0, 5
/* 000014 2441001F */  addiu $at, $v0, 0x1f
/* 000018 00017143 */  sra   $t6, $at, 5
.L001C:
/* 00001C 03E00008 */  jr    $ra
/* 000020 01C01025 */   move  $v0, $t6
