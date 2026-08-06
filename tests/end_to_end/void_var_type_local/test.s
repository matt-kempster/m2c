.set noat
.set noreorder

glabel test
/* 000000 00400000 27BDFFE0 */  addiu $sp, $sp, -0x20
/* 000004 00400004 3C018000 */  lui   $at, 0x8000
/* 000008 00400008 8C210000 */  lw    $at, 0($at)       # Load pointer from global
/* 00000C 0040000C AFA10018 */  sw    $at, 0x18($sp)    # Store to sp18
/* 000010 00400010 8FA20018 */  lw    $v0, 0x18($sp)    # Load sp18
/* 000014 00400014 8C420000 */  lw    $v0, 0($v0)       # Load sp18->field_0x00
/* 000018 00400018 03E00008 */  jr    $ra
/* 00001C 0040001C 27BD0020 */   addiu $sp, $sp, 0x20
