.set noat      # allow manual use of $at
.set noreorder # don't insert nops after branches

glabel test
    break 1
    break 2
    badinstr $t0, $t0
    badinstr2 $t1, $t1
    badinstr3 $v0, $t2
    sllv $t1, $t1, $t1
    sw $t1, ($zero)
    jr $ra
     nop
