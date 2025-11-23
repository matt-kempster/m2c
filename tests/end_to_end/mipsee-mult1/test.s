glabel test
multu $zero, $a0, $a1
multu1 $zero, $a2, $a3
mflo $t0
mflo1 $t1
addu $v0, $t0, $t1
mfhi $t0
mfhi1 $t1
addu $v0, $v0, $t0
addu $v0, $v0, $t1
jr $ra
 nop

glabel test2
mult $zero, $a0, $a1
mult1 $t2, $a2, $a3
mflo $t0
mflo1 $t1
addu $v0, $t0, $t1
addu $v0, $v0, $t2
jr $ra
 nop

glabel test3
mult $zero, $a0, $a0
mtlo1 $a1
mthi1 $a2
madd1 $t2, $a2, $a3
maddu1 $t2, $a2, $a3
div1 $t2, $t2, $a0
divu1 $zero, $t2, $a0
mflo1 $t0
mfhi1 $t1
addu $v0, $t0, $t1
mflo $t0
mfhi $t1
addu $v0, $v0, $t0
addu $v0, $v0, $t1
jr $ra
 nop
