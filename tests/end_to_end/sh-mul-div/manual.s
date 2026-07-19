    .global _test
_test:
    muls.w r5,r4
    rts
    sts macl,r0

    .global _test_mulu
_test_mulu:
    mulu.w r5,r4
    rts
    sts macl,r0

    .global _test_delay
_test_delay:
    sts.l pr,@-r15
    mov.l L_div,r1
    jsr @r1
    mov r6,r5
    lds.l @r15+,pr
    rts
    nop
L_div:
    .long ___sdivsi3
