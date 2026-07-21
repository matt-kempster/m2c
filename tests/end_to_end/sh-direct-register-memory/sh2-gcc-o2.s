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
	mov.b	@r4,r0
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadw
_test_loadw:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.w	@r4,r0
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadl
_test_loadl:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r4,r0
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storeb
_test_storeb:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.b	r5,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storew
_test_storew:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.w	r5,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storel
_test_storel:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	r5,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadb_postinc
_test_loadb_postinc:
	mov.l	r14,@-r15
	mov.l	@r4,r1
	mov	r15,r14
	mov.b	@r1+,r0
	mov.l	r1,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadw_postinc
_test_loadw_postinc:
	mov.l	r14,@-r15
	mov.l	@r4,r1
	mov	r15,r14
	mov.w	@r1+,r0
	mov.l	r1,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_loadl_postinc
_test_loadl_postinc:
	mov.l	r14,@-r15
	mov.l	@r4,r1
	mov	r15,r14
	mov.l	@r1+,r0
	mov.l	r1,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storeb_predec
_test_storeb_predec:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r4,r1
	mov.b	r5,@-r1
	mov.l	r1,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storew_predec
_test_storew_predec:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r4,r1
	mov.w	r5,@-r1
	mov.l	r1,@r4
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_storel_predec
_test_storel_predec:
	mov.l	r14,@-r15
	mov	r15,r14
	mov.l	@r4,r1
	mov.l	r5,@-r1
	mov.l	r1,@r4
	rts
	mov.l	@r15+,r14
