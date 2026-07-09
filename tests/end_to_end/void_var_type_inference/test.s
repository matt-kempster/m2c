.set noat
.set noreorder

glabel test
/* 000000 00400000 8C820000 */  lw    $v0, 0($a0)   # arg0->first_field
/* 000004 00400004 8C830004 */  lw    $v1, 4($a0)   # arg0->second_field
/* 000008 00400008 8C840008 */  lw    $a0, 8($a0)   # arg0->third_field
/* 00000C 0040000C 00431021 */  addu  $v0, $v0, $v1
/* 000010 00400010 00441021 */  addu  $v0, $v0, $a0
/* 000014 00400014 03E00008 */  jr    $ra
/* 000018 00400018 00000000 */   nop
