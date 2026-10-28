.set noat
.set noreorder

glabel test
/* 000000 00400000 3C088000 */  lui   $t0, 0x8000        # Load the address of a global pointer
/* 000004 00400004 8D080000 */  lw    $t0, 0($t0)        # Load the pointer itself
/* 000008 00400008 8D020004 */  lw    $v0, 4($t0)        # Load field at offset 4
/* 00000C 0040000C 8D030000 */  lw    $v1, 0($t0)        # Load field at offset 0
/* 000010 00400010 00431021 */  addu  $v0, $v0, $v1      # Sum both fields
/* 000014 00400014 03E00008 */  jr    $ra
/* 000018 00400018 00000000 */   nop
