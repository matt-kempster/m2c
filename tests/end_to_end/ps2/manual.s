glabel test
addiu   $sp,$sp,-48
sq      $ra,32($sp)
move    $a0,$sp
ori     $a1,$sp,0x4
li      $t4,2
mult    $a2,$a2,$t4
mtlo    $a3
madd    $a2,$t4
jal     foo
 mflo    $a3
lq      $ra,32($sp)
jr      $ra
addiu   $sp,$sp,48
