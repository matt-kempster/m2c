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
	mov	r4,r0
	cmp/eq	#1,r0
	bt.s	L4
	mov	r15,r14
	mov	#1,r1
	cmp/gt	r1,r0
	bt.s	L8
	cmp/eq	#2,r0
	tst	r0,r0
	bt.s	L9
	mov	#11,r0
	bra	L9
	mov	#-7,r0
L8:
	bt.s	L9
	mov	#19,r0
	bra	L6
	mov	#-7,r0
L4:
	mov	#42,r0
L6:
L9:
	mov	r14,r15
	rts
	mov.l	@r15+,r14
