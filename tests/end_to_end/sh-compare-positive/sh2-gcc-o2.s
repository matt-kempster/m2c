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
	cmp/pz	r4
	bt.s	L2
	mov	r15,r14
	bra	L4
	mov	#-1,r0
L2:
	cmp/pl	r4
	bt.s	L3
	mov	#1,r0
	mov	#0,r0
L3:
L4:
	mov	r14,r15
	rts
	mov.l	@r15+,r14
