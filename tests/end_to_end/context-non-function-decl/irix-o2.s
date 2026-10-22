.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches


glabel test
/* 0000B0 004000B0 03E00008 */  jr    $ra
/* 0000B4 004000B4 24820001 */   addiu $v0, $a0, 1
