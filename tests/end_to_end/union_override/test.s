.set noat
.set noreorder

glabel test
/* 000000 00400000 8C820000 */  lw    $v0, 0($a0)   # Load from struct->data (offset 0)
/* 000004 00400004 03E00008 */  jr    $ra
/* 000008 00400008 AC850000 */   sw    $a1, 0($a0)  # Store to struct->data (offset 0)
