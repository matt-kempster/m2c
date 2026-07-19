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
	mov	r15,r14
	mul.l	r5,r4
	mov.l	@r15+,r14
	rts
	sts	macl,r0
	.align 2
	.global	_test_umul
_test_umul:
	mov.l	r14,@-r15
	mov	r15,r14
	mul.l	r5,r4
	mov.l	@r15+,r14
	rts
	sts	macl,r0
	.align 2
	.global	_test_mulsw
_test_mulsw:
	mov.l	r14,@-r15
	mov	r15,r14
	exts.w	r4,r4
	exts.w	r5,r5
	muls	r5,r4
	mov.l	@r15+,r14
	rts
	sts	macl,r0
	.align 2
	.global	_test_muluw
_test_muluw:
	mov.l	r14,@-r15
	mov	r15,r14
	extu.w	r4,r4
	extu.w	r5,r5
	mulu	r5,r4
	mov.l	@r15+,r14
	rts
	sts	macl,r0
	.align 2
	.global	_test_sdiv
_test_sdiv:
	mov.l	r14,@-r15
	sts.l	pr,@-r15
	mov.l	L6,r7
	jsr	@r7
	mov	r15,r14
	lds.l	@r15+,pr
	rts
	mov.l	@r15+,r14
L7:
	.align 2
L6:
	.long	___sdivsi3
	.align 2
	.global	_test_udiv
_test_udiv:
	mov.l	r14,@-r15
	sts.l	pr,@-r15
	mov.l	L9,r1
	jsr	@r1
	mov	r15,r14
	lds.l	@r15+,pr
	rts
	mov.l	@r15+,r14
L10:
	.align 2
L9:
	.long	___udivsi3
