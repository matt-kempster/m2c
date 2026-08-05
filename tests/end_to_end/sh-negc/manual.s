.global _test
_test:
    clrt
    negc r4,r0
    rts
    nop

.global _test_set
_test_set:
    sett
    negc r4,r0
    rts
    nop

.global _test_chain
_test_chain:
    clrt
    negc r4,r4
    negc r5,r5
    rts
    mov r5,r0

.global _test_set_chain
_test_set_chain:
    sett
    negc r4,r4
    negc r5,r5
    rts
    mov r5,r0
