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
	mov.l	L2,r2
	mov.l	r4,@r2
	mov	r4,r3
	add	r3,r3
	mov.l	r3,@r2
	add	r4,r3
	mov.l	r3,@r2
	mov	r4,r7
	shll2	r7
	mov.l	r7,@r2
	add	r4,r7
	mov.l	r7,@r2
	mov	r3,r1
	add	r1,r1
	mov.l	r1,@r2
	mov	#7,r1
	mul.l	r1,r4
	sts	macl,r6
	mov.l	r6,@r2
	mov	r4,r1
	shll2	r1
	add	r1,r1
	mov.l	r1,@r2
	mov	#9,r1
	mul.l	r1,r4
	sts	macl,r6
	mov.l	r6,@r2
	mov	r7,r1
	add	r1,r1
	mov.l	r1,@r2
	mov	#11,r1
	mul.l	r1,r4
	sts	macl,r6
	mov	#13,r1
	mul.l	r1,r4
	shll2	r3
	mov.l	r6,@r2
	sts	macl,r6
	mov	#14,r1
	mul.l	r1,r4
	mov.l	r3,@r2
	mov.l	r6,@r2
	sts	macl,r6
	mov	#15,r1
	mul.l	r1,r4
	mov.l	r6,@r2
	sts	macl,r6
	mov.l	r6,@r2
	mov	r4,r1
	shll2	r1
	shll2	r1
	mov.l	r1,@r2
	mov	#17,r1
	mul.l	r1,r4
	sts	macl,r6
	mov	#18,r1
	mul.l	r1,r4
	mov	r15,r14
	mov.l	r6,@r2
	sts	macl,r6
	mov	#19,r1
	mul.l	r1,r4
	mov.l	r6,@r2
	sts	macl,r6
	mov	#21,r1
	mul.l	r1,r4
	shll2	r7
	mov.l	r6,@r2
	sts	macl,r6
	mov	#22,r1
	mul.l	r1,r4
	mov.l	r7,@r2
	mov.l	r6,@r2
	sts	macl,r6
	mov.l	r6,@r2
	rts
	mov.l	@r15+,r14
L3:
	.align 2
L2:
	.long	_y
	.comm _y,4
