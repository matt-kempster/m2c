glabel test
  addiu $sp, $sp, -0x110
  jal   foo
   nop
  move  $s5, $v0
  lbu   $t0, ($zero)
  bnez  $t0, .L0043D650
   nop
  sw    $s5, 0x10c($sp)
  lw    $s5, 0x10c($sp)
.L0043D650:
  sb    $zero, 3($s5)
  sb    $zero, 4($s5)
  lw    $ra, 0x3c($sp)
  jr    $ra
   addiu $sp, $sp, 0x110
