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
	add	#3,r4
	mov.b	@r4,r0
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadw_disp
_test_loadw_disp:
	mov.l	r14,@-r15
	mov	r15,r14
	add	#6,r4
	mov.w	@r4,r0
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadl_disp
_test_loadl_disp:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@(12,r4),r0
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storeb_disp
_test_storeb_disp:
	mov.l	r14,@-r15
	mov	r15,r14
	add	#3,r4
	mov.b	r5,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storew_disp
_test_storew_disp:
	mov.l	r14,@-r15
	mov	r15,r14
	add	#6,r4
	mov.w	r5,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storel_disp
_test_storel_disp:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	r5,@(12,r4)
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadb_indexed
_test_loadb_indexed:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r5,r0
	mov.b	@(r0,r4),r0
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadw_indexed
_test_loadw_indexed:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r5,r0
	add	r0,r0
	mov.w	@(r0,r4),r0
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadl_indexed
_test_loadl_indexed:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r5,r0
	shll2	r0
	mov.l	@(r0,r4),r0
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storeb_indexed
_test_storeb_indexed:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r5,r0
	mov.b	r6,@(r0,r4)
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storew_indexed
_test_storew_indexed:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r5,r0
	add	r0,r0
	mov.w	r6,@(r0,r4)
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storel_indexed
_test_storel_indexed:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r5,r0
	shll2	r0
	mov.l	r6,@(r0,r4)
	rts
	mov.l	@r15+,r14
