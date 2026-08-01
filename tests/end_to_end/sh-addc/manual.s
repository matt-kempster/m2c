.global _test
_test:
    clrt
    addc r5,r4
    rts
    mov r4,r0

.global _test_set
_test_set:
    sett
    addc r5,r4
    rts
    mov r4,r0

.global _test_chain
_test_chain:
    clrt
    addc r6,r4
    addc r7,r5
    rts
    mov r5,r0
