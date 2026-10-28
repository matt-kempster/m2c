.set noat
.set noreorder

glabel test
/* 000000 00400000 8C820000 */  lw    $v0, 0($a0)   # Load from arg0->field_0x00
/* 000004 00400004 8C830004 */  lw    $v1, 4($a0)   # Load from arg0->field_0x04
/* 000008 00400008 00431021 */  addu  $v0, $v0, $v1 # Add them
/* 00000C 0040000C 03E00008 */  jr    $ra
/* 000010 00400010 00000000 */   nop
