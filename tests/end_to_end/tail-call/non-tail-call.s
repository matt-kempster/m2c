test:
addiu       $sp, $sp, -0x10
sd          $ra, 0x0($sp)
jal         foo
 nop
ld          $ra, 0x0($sp)
j           loc_bar
 addiu      $sp, $sp, 0x10
loc_bar:
jr $ra
 nop
