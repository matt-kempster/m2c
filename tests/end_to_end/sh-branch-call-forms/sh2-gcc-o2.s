	.file	"input.i"
	.data

! Hitachi SH cc1 (cygnus-2.7-96q3 SOA-960904) arguments: -O -fdefer-pop
! -fcse-follow-jumps -fcse-skip-blocks -fexpensive-optimizations
! -fthread-jumps -fstrength-reduce -fpeephole -fforce-mem -ffunction-cse
! -finline -fkeep-static-consts -fcaller-saves -freg-struct-return
! -fdelayed-branch -frerun-cse-after-loop -fschedule-insns2 -fcommon
! -fgnu-linker -m2

gcc2_compiled.:
___gnu_compiled_c:
	.text
	.align 2
	.global	_test
_test:
	mov.l	r14,@-r15
	tst	r4,r4
	bt.s	L2
	mov	r15,r14
	bra	L3
	mov	#2,r0
L2:
	mov	#1,r0
L3:
	mov	r14,r15
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_if_nonzero
_test_if_nonzero:
	mov.l	r14,@-r15
	tst	r4,r4
	bf.s	L5
	mov	r15,r14
	bra	L6
	mov	#4,r0
L5:
	mov	#3,r0
L6:
	mov	r14,r15
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_if_greater
_test_if_greater:
	mov.l	r14,@-r15
	mov	r4,r1
	mov	r5,r0
	cmp/gt	r0,r1
	bf.s	L9
	mov	r15,r14
	mov	r1,r0
L9:
	mov	r14,r15
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loop
_test_loop:
	mov.l	r14,@-r15
	mov	r15,r14
	tst	r4,r4
	bt.s	L12
	mov	#0,r0
	add	r4,r0
L15:
	dt	r4
	bf.s	L15
	add	r4,r0
	sub	r4,r0
L12:
	mov	r14,r15
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_call
_test_call:
	mov.l	r14,@-r15
	sts.l	pr,@-r15
	mov.l	L17,r0
	jsr	@r0
	mov	r15,r14
	mov	r14,r15
	lds.l	@r15+,pr
	mov.l	@r15+,r14
	rts
	add	#1,r0
L18:
	.align 2
L17:
	.long	_callee
