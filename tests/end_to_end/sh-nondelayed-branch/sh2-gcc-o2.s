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
	tst	r4,r4
	bt.s	L2
	mov	r15,r14
	mov.l	r4,@r6
L2:
	tst	r5,r5
	bt	L3
	mov.l	r5,@r6
L3:
	mov	r14,r15
	rts
	mov.l	@r15+,r14
	.align 2
	.global	_test_bf
_test_bf:
	mov.l	r14,@-r15
	tst	r4,r4
	bf.s	L5
	mov	r15,r14
	mov.l	r4,@r6
L5:
	tst	r5,r5
	bf	L6
	mov.l	r5,@r6
L6:
	mov	r14,r15
	rts
	mov.l	@r15+,r14
