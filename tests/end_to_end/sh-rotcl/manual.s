.global _test
_test:
    clrt
    rotcl r4
    rts
    mov r4,r0

.global _test_set
_test_set:
    sett
    rotcl r4
    rts
    mov r4,r0

.global _test_chain
_test_chain:
    clrt
    rotcl r4
    rotcl r5
    rts
    mov r5,r0

.global _test_rotate_low
_test_rotate_low:
    mov r5,r0
    shll r0
    rotcl r4
    rotcl r5
    rts
    mov r4,r0
