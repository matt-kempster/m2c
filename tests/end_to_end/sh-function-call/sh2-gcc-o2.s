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
	.global	test
test:
	mov.l	r14,@-r15
	sts.l	pr,@-r15
	mov	r15,r14
	mov	r4,r5
	mov.l	L2,r0
	jsr	@r0
	add	#1,r5
	mov	r14,r15
	lds.l	@r15+,pr
	mov.l	@r15+,r14
	rts
	add	#3,r0
L3:
	.align 2
L2:
	.long	_callee
