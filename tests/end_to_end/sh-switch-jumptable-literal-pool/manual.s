glabel test
mov r4,r1
add r1,r1
mova L1,r0
mov.w @(r0,r1),r1
add r1,r0
bra L2
nop
L3:
.long 0
L2:
jmp @r0
nop
L1:
.word L4-L1
L4:
rts
mov #1,r0
