glabel test
addiu    $sp, $sp, -0x30
sq       $ra, 0x20($sp)
sq       $s1, 0x10($sp)
sq       $s0, 0x00($sp)
paddub   $s0, $a0, $zero
jal      func__Fv
 paddub  $s1, $a1, $zero
paddub   $a0, $s0, $zero
jal      func2__FiPPc
 paddub  $a1, $s1, $zero
lq       $s0, 0x00($sp)
lq       $s1, 0x10($sp)
lq       $ra, 0x20($sp)
jr       $ra
 addiu   $sp, $sp, -0x30
