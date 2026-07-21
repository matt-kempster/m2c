    .file   "input.s"
    .text
    .align  2
    .global _test
_test:
    mov.l   @r4, r1
    mov.l   @r1+, r0
    mov.l   r1, @r4
    rts
    nop
    .align  2
    .global _test_storel_predec
_test_storel_predec:
    mov.l   @r4, r1
    mov.l   r5, @-r1
    mov.l   r1, @r4
    rts
    nop
    .align  2
    .global _test_loadb_predec
_test_loadb_predec:
    mov.l   @r4, r1
    mov.b   @-r1, r0
    mov.l   r1, @r4
    rts
    nop
    .align  2
    .global _test_loadw_predec
_test_loadw_predec:
    mov.l   @r4, r1
    mov.w   @-r1, r0
    mov.l   r1, @r4
    rts
    nop
