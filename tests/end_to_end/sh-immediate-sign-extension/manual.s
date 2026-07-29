glabel test
mov #255, r0
rts
nop

glabel test_min
mov #128, r0
rts
nop

glabel test_add
mov r4, r0
rts
add #255, r0

glabel test_compare
cmp/eq #255, r4
movt r0
rts
nop
