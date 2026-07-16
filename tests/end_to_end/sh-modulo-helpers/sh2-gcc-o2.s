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
	mov.l	r8,@-r15
	mov.l	r14,@-r15
	sts.l	pr,@-r15
	mov	r15,r14
	mov	r4,r8
	mov.l	L2,r6
	jsr	@r6
	mov	r5,r7
	mul.l	r7,r0
	lds.l	@r15+,pr
	mov.l	@r15+,r14
	sts	macl,r1
	sub	r1,r8
	mov	r8,r0
	rts
	mov.l	@r15+,r8
L3:
	.align 2
L2:
	.long	___sdivsi3
	.align 2
	.global	_test_unsigned
_test_unsigned:
	mov.l	r14,@-r15
	sts.l	pr,@-r15
	mov	r15,r14
	mov	r4,r3
	mov.l	L5,r2
	jsr	@r2
	mov	r5,r1
	mul.l	r1,r0
	lds.l	@r15+,pr
	mov.l	@r15+,r14
	sts	macl,r1
	sub	r1,r3
	rts
	mov	r3,r0
L6:
	.align 2
L5:
	.long	___udivsi3
