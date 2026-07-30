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
	dmuls.l	r5,r4
	mov.l	@r15+,r14
	sts	mach,r3
	sts	macl,r4
	sts	mach,r2
	mov	r3,r1
	shll	r1
	subc	r1,r1
	rts
	mov	r2,r0
	.align 2
	.global	_test_unsigned
_test_unsigned:
	mov.l	r14,@-r15
	mov	r15,r14
	dmulu.l	r5,r4
	mov.l	@r15+,r14
	sts	mach,r3
	sts	macl,r4
	mov	r3,r2
	nop
	mov	#0,r1
	rts
	mov	r2,r0
