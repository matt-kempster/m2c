glabel test

lw    $t1, %got(local_sym)($gp)
addiu $t1, $t1, %lo(local_sym)
sw    $zero, 8($t1)

lw    $t1, %got(global_sym)($gp)
sw    $zero, 8($t1)

lw    $t9, %got(func)($gp)
jalr  $t9
 nop

jr $ra
 nop
