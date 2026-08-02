.global _test
_test:
    clrt
    subc r5,r4
    rts
    mov r4,r0

.global _test_set
_test_set:
    sett
    subc r5,r4
    rts
    mov r4,r0

.global _test_chain
_test_chain:
    clrt
    subc r6,r4
    subc r7,r5
    rts
    mov r5,r0
