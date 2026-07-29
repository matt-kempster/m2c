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
	mov.l	@r15+,r14
	rts
	mov	#-1,r0
	.align 2
	.global	_test_min
_test_min:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r15+,r14
	rts
	mov	#-128,r0
	.align 2
	.global	_test_max
_test_max:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r15+,r14
	rts
	mov	#127,r0
	.align 2
	.global	_test_add
_test_add:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	add	#-1,r0
	.align 2
	.global	_test_compare
_test_compare:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	cmp/eq	#-1,r0
	mov.l	@r15+,r14
	rts
	movt	r0
