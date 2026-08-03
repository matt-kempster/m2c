glabel test
mov.l r13,@-r15
mov.l L1,r13
jmp @r13
mov.l @r15+,r13
L1:
.long L2
L2:
rts
mov #1,r0
