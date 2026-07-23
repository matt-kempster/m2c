	.file	"manual.s"
	.text
	.align 2

! The quotient is used straight out of r0, with no intervening move, and the
! remainder feeds an add rather than being returned.
	.global	_test
_test:
	sts.l	pr,@-r15
	mov.l	r8,@-r15
	mov	r4,r8
	mov	r5,r6
	mov	r8,r4
	mov.l	L100,r7
	jsr	@r7
	mov	r6,r5
	mul.l	r6,r0
	sts	macl,r1
	sub	r1,r8
	add	#1,r8
	mov	r8,r0
	mov.l	@r15+,r8
	lds.l	@r15+,pr
	rts
	nop
	.align 2
L100:
	.long	___sdivsi3

	.align 2
! An `extu.w` sits between the literal pool load and the jsr, so even
	.global	_mod_extu_delay
_mod_extu_delay:
	sts.l	pr,@-r15
	mov	r4,r7
	mov	r5,r1
	mov	r7,r4
	mov.l	L101,r0
	extu.w	r1,r6
	jsr	@r0
	mov	r6,r5
	mov	r0,r5
	mul.l	r6,r5
	sts	macl,r1
	sub	r1,r7
	mov	r7,r0
	lds.l	@r15+,pr
	rts
	nop
	.align 2
L101:
	.long	___sdivsi3

	.align 2
! The dividend is loaded into r8 *after* the multiply, between `mul.l` and
! `sts`, so the run is not contiguous
	.global	_mod_split_dividend
_mod_split_dividend:
	sts.l	pr,@-r15
	mov.l	r8,@-r15
	mov.l	r11,@-r15
	mov	r4,r11
	mov	r5,r6
	mov	r11,r4
	mov.l	L102,r0
	extu.w	r6,r6
	jsr	@r0
	mov	r6,r5
	mov	r0,r7
	mul.l	r6,r7
	mov	r11,r8
	sts	macl,r1
	sub	r1,r8
	mov	r8,r0
	mov.l	@r15+,r11
	mov.l	@r15+,r8
	lds.l	@r15+,pr
	rts
	nop
	.align 2
L102:
	.long	___sdivsi3

	.align 2
! The remainder feeds a further multiply instead of being returned.
	.global	_mod_feeds_multiply
_mod_feeds_multiply:
	sts.l	pr,@-r15
	mov.l	r13,@-r15
	mov	r4,r13
	mov	r5,r7
	mov	r13,r4
	mov.l	L103,r0
	extu.w	r7,r7
	jsr	@r0
	mov	r7,r5
	mov	r0,r5
	mul.l	r7,r5
	mov	r13,r4
	sts	macl,r1
	sub	r1,r4
	mov	r4,r1
	mul.l	r13,r1
	sts	macl,r0
	mov.l	@r15+,r13
	lds.l	@r15+,pr
	rts
	nop
	.align 2
L103:
	.long	___sdivsi3

	.align 2
! quotient moved out of r0 before the multiply.
	.global	_mod_moved_quotient
_mod_moved_quotient:
	sts.l	pr,@-r15
	mov.l	r9,@-r15
	mov	r4,r7
	mov	r5,r9
	mov	r7,r4
	mov.l	L104,r0
	jsr	@r0
	mov	r9,r5
	mov	r0,r5
	mul.l	r9,r5
	sts	macl,r1
	sub	r1,r7
	mov	r7,r0
	add	#1,r0
	mov.l	@r15+,r9
	lds.l	@r15+,pr
	rts
	nop
	.align 2
L104:
	.long	___udivsi3

	.align 2
! Two modulo sequences interleaved by the scheduler: the `sub` completing the
! first one is placed in the delay slot of the second one's call.
	.global	_mod_interleaved
_mod_interleaved:
	sts.l	pr,@-r15
	mov.l	r9,@-r15
	mov.l	r11,@-r15
	mov.l	r13,@-r15
	mov	r4,r13
	mov	r5,r9
	mov	r4,r11
	mov	r13,r4
	mov.l	L105,r0
	jsr	@r0
	mov	r13,r5
	mov	r0,r7
	mov	r13,r4
	mul.l	r13,r7
	mov	r9,r5
	mov.l	L105,r0
	sts	macl,r1
	mov	r11,r7
	jsr	@r0
	sub	r1,r7
	mov	r0,r5
	mul.l	r9,r5
	mov	r13,r0
	sts	macl,r1
	sub	r1,r0
	mov.l	@r15+,r13
	mov.l	@r15+,r11
	mov.l	@r15+,r9
	lds.l	@r15+,pr
	rts
	nop
	.align 2
L105:
	.long	___udivsi3
