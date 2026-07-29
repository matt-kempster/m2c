glabel test
mova L1, r0
mov.l @r0, r0
rts
nop
.align 2
L1:
.long 0
