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
	mov.l	r8,@-r15
	mov.l	r14,@-r15
	sts.l	pr,@-r15
	mov	r15,r14
	mov	r5,r8
	mov	#0,r6
	mov	#1,r3
	mov	#3,r5
	mov	#4,r7
L5:
	mov	r6,r1
	mov	r7,r0
	mov.l	@(r0,r4),r2
	shll2	r1
	mov	r1,r0
	mov.l	@(r0,r4),r1
	cmp/gt	r1,r2
	bf	L4
	mov	r3,r6
L4:
	add	#1,r3
	cmp/gt	r5,r3
	bf.s	L5
	add	#4,r7
	mov	#3,r1
	cmp/hi	r1,r6
	bt.s	L8
	mov	r6,r1
	add	r1,r1
	mova	L13,r0
	mov.w	@(r0,r1),r1
	add	r1,r0
	jmp        @r0
	nop
	.align 2
L13:
	.word	L9-L13
	.word	L10-L13
	.word	L11-L13
	.word	L12-L13
L9:
	bra	L8
	add	r8,r8
L10:
	bra	L8
	add	#3,r8
L11:
	mov.l	L15,r0
	jsr	@r0
	mov	r8,r4
	bra	L8
	add	r0,r8
L12:
	mov.l	L15,r0
	jsr	@r0
	mov	r8,r4
	sub	r0,r8
L8:
	mov	r14,r15
	lds.l	@r15+,pr
	mov.l	@r15+,r14
	mov	r8,r0
	rts
	mov.l	@r15+,r8
L16:
	.align 2
L15:
	.long	_sink
