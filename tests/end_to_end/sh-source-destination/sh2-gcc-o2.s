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
	and	r5,r0
	.align 2
	.global	_test_or
_test_or:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	or	r5,r0
	.align 2
	.global	_test_xor
_test_xor:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	xor	r5,r0
	.align 2
	.global	_test_not
_test_not:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r15+,r14
	rts
	not	r4,r0
	.align 2
	.global	_test_neg
_test_neg:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r15+,r14
	rts
	neg	r4,r0
	.align 2
	.global	_test_eq
_test_eq:
	mov.l	r14,@-r15
	mov	r15,r14
	cmp/eq	r5,r4
	mov.l	@r15+,r14
	rts
	movt	r0
	.align 2
	.global	_test_ge
_test_ge:
	mov.l	r14,@-r15
	mov	r15,r14
	cmp/ge	r5,r4
	mov.l	@r15+,r14
	rts
	movt	r0
	.align 2
	.global	_test_gt
_test_gt:
	mov.l	r14,@-r15
	mov	r15,r14
	cmp/gt	r5,r4
	mov.l	@r15+,r14
	rts
	movt	r0
	.align 2
	.global	_test_hs
_test_hs:
	mov.l	r14,@-r15
	mov	r15,r14
	cmp/hs	r5,r4
	mov.l	@r15+,r14
	rts
	movt	r0
	.align 2
	.global	_test_hi
_test_hi:
	mov.l	r14,@-r15
	mov	r15,r14
	cmp/hi	r5,r4
	mov.l	@r15+,r14
	rts
	movt	r0
	.align 2
	.global	_test_extsb
_test_extsb:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r15+,r14
	rts
	exts.b	r4,r0
	.align 2
	.global	_test_extsw
_test_extsw:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r15+,r14
	rts
	exts.w	r4,r0
	.align 2
	.global	_test_extub
_test_extub:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r15+,r14
	rts
	extu.b	r4,r0
	.align 2
	.global	_test_extuw
_test_extuw:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r15+,r14
	rts
	extu.w	r4,r0
	.align 2
	.global	_test_swapw
_test_swapw:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r15+,r14
	rts
	swap.w	r4,r0
