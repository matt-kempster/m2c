glabel test
li $a0, sym1
sw $a0, ($zero)
sw $zero, sym2($zero)
.set sym3, 0x20
sw $zero, sym3($zero)
sw $zero, (sym3 * 2)($zero)
li $a0, ((sym3 * 2) + 0x10)
sw $a0, ($zero)
jr $ra
nop
