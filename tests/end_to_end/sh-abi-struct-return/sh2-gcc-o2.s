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
	add	#-12,r15
	mov	r15,r14
	mov	r2,r0
	mov.l	r4,@r14
	mov	r4,r1
	add	#1,r1
	mov.l	r1,@(4,r14)
	mov	r4,r1
	add	#2,r1
	mov.l	r1,@(8,r14)
	mov.l	r4,@r0
	mov.l	@(4,r14),r1
	mov.l	r1,@(4,r0)
	mov.l	@(8,r14),r1
	mov.l	r1,@(8,r0)
	add	#12,r14
	mov	r14,r15
	rts
	mov.l	@r15+,r14
