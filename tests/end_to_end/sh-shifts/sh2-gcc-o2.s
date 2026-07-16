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
	add	r0,r0
	.align 2
	.global	_test_shal
_test_shal:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	add	r0,r0
	.align 2
	.global	_test_shlr
_test_shlr:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	shlr	r0
	.align 2
	.global	_test_shar
_test_shar:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	shar	r0
	.align 2
	.global	_test_shll2
_test_shll2:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	shll2	r0
	.align 2
	.global	_test_shlr2
_test_shlr2:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	shlr2	r0
	.align 2
	.global	_test_shll8
_test_shll8:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	shll8	r0
	.align 2
	.global	_test_shlr8
_test_shlr8:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	shlr8	r0
	.align 2
	.global	_test_shll16
_test_shll16:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	shll16	r0
	.align 2
	.global	_test_shlr16
_test_shlr16:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	shlr16	r0
	.align 2
	.global	_test_rotl
_test_rotl:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	rotl	r0
	.align 2
	.global	_test_rotr
_test_rotr:
	mov.l	r14,@-r15
	mov	r15,r14
	mov	r4,r0
	mov.l	@r15+,r14
	rts
	rotr	r0
