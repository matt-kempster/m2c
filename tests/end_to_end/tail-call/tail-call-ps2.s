test:
addiu       $sp, $sp, -0x10
sd          $ra, 0x0($sp)
jal         foo
 nop
ld          $ra, 0x0($sp)
j           bar
 addiu      $sp, $sp, 0x10
nop
