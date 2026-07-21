	.file	"input.s"
	.global	_test
	.text
	.align 2
_test:
	mov.l	r14,@-r15
	sts.l	pr,@-r15
	mov	r15,r14
	mov	#1,r4
	mov	#2,r5
	mov	#3,r6
	mov	#4,r7
	mov	#5,r8
	mov.l	r8,@-r15
	mov.l	L1,r0
	jsr	@r0
	nop
	mov.l	@r15+,r14
	lds.l	@r15+,pr
	rts
	nop
	.align 2
L1:
	.long	_callee
