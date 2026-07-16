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
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	and	#90,r0
	.align 2
	.global	_test_or
_test_or:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	or	#90,r0
	.align 2
	.global	_test_xor
_test_xor:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	xor	#90,r0
	.align 2
	.global	_test_tst
_test_tst:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	and	#90,r0
	tst	r0,r0
	mov.l	@r15+,r14
	rts
	movt	r0
