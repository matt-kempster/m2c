glabel test
addiu $sp, $sp, -0x20
sw $ra, 0x1c($sp)
move $t0, $sp
li $v0, 5
sw $v0, 0x10($sp)
li $a0, 1
li $a1, 2
li $a2, 3
jal writer
 li $a3, 4
lw $ra, 0x1c($sp)
jr $ra
 addiu $sp, $sp, 0x20
