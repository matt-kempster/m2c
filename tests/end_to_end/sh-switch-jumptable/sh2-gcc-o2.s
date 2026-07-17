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
	mov	#3,r1
	cmp/hi	r1,r4
	bt.s	L7
	mov	r15,r14
	mov	r4,r1
	add	r1,r1
	mova	L8,r0
	mov.w	@(r0,r1),r1
	add	r1,r0
	jmp        @r0
	nop
	.align 2
L8:
	.word	L3-L8
	.word	L4-L8
	.word	L5-L8
	.word	L6-L8
L3:
	bra	L9
	mov	#11,r0
L4:
	bra	L9
	mov	#42,r0
L5:
	bra	L9
	mov	#19,r0
L6:
	bra	L9
	mov	#73,r0
L7:
	mov	#-7,r0
L9:
	mov	r14,r15
	rts
	mov.l	@r15+,r14
