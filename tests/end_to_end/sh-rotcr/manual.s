.global _test
_test:
    clrt
    rotcr r4
    rts
    mov r4,r0

.global _test_set
_test_set:
    sett
    rotcr r4
    rts
    mov r4,r0

.global _test_chain
_test_chain:
    clrt
    rotcr r5
    rotcr r4
    rts
    mov r4,r0

.global _test_rotate_high
_test_rotate_high:
    mov r4,r0
    shlr r0
    rotcr r5
    rotcr r4
    rts
    mov r5,r0
