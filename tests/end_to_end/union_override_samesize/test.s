.set noat
.set noreorder

glabel test_func
/* 000000 00400000 8C820000 */  lw    $v0, 0($a0)   # Load first field
/* 000004 00400004 8C830004 */  lw    $v1, 4($a0)   # Load second field
/* 000008 00400008 03E00008 */  jr    $ra
/* 00000C 0040000C 00000000 */   nop
