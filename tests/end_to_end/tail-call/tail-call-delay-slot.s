test:
lui         $v0, %hi(D_410000)
addiu       $v0, $v0, %lo(D_410000)
j           bar
 addu       $a0, $a0, $v0
nop
