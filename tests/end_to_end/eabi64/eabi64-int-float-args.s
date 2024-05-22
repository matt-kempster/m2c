glabel test
  addiu       $sp, $sp, -0x10
  sq          $ra, 0x0($sp)
  jal         no_args_func
   nop
  lui         $at, (0x40000000 >> 16)
  mtc1        $at, $f12
  addiu       $a0, $zero, 0x1
  lui         $at, (0x3F800000 >> 16)
  mtc1        $at, $f13
  jal         func_with_args
   addiu      $a1, $zero, 0x2
  lq          $ra, 0x0($sp)
  jr          $ra
   addiu      $sp, $sp, 0x10
