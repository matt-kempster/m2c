
.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches


glabel test
/* 000090 00400090 27BDFFF8 */  addiu $sp, $sp, -8
/* 000094 00400094 AFBF0000 */  sw    $ra, 0($sp)
/* 000098 00400098 AFA40008 */  sw    $a0, 8($sp)
/* 00009C 0040009C 8FA20008 */  lw    $v0, 8($sp)
/* 0000A0 004000A0 03E00008 */  jr    $ra
/* 0000A4 004000A4 27BD0008 */   addiu $sp, $sp, 8
