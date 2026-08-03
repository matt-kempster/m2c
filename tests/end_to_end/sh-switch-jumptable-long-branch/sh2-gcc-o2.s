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
	mov.l	L50,r8
	jsr	@r8
	mov	r4,r0
	and	#127,r0
	mov	#43,r1
	cmp/hi	r1,r0
	bf.s	LF100
	mov	r0,r1
	bra	L47
	nop
LF100:
	add	r1,r1
	mova	L48,r0
	mov.w	@(r0,r1),r1
	add	r1,r0
	jmp        @r0
	nop
	.align 2
L48:
	.word	L3-L48
	.word	L4-L48
	.word	L5-L48
	.word	L6-L48
	.word	L7-L48
	.word	L8-L48
	.word	L9-L48
	.word	L10-L48
	.word	L11-L48
	.word	L12-L48
	.word	L13-L48
	.word	L14-L48
	.word	L15-L48
	.word	L16-L48
	.word	L17-L48
	.word	L18-L48
	.word	L19-L48
	.word	L20-L48
	.word	L21-L48
	.word	L22-L48
	.word	L23-L48
	.word	L24-L48
	.word	L25-L48
	.word	L26-L48
	.word	L27-L48
	.word	L28-L48
	.word	L29-L48
	.word	L30-L48
	.word	L31-L48
	.word	L32-L48
	.word	L33-L48
	.word	L34-L48
	.word	L35-L48
	.word	L36-L48
	.word	L37-L48
	.word	L38-L48
	.word	L39-L48
	.word	L40-L48
	.word	L41-L48
	.word	L42-L48
	.word	L43-L48
	.word	L44-L48
	.word	L45-L48
	.word	L46-L48
L3:
	bra	L49
	mov	#11,r0
L4:
	bra	L49
	mov	#48,r0
L5:
	bra	L49
	mov	#85,r0
L6:
	bra	L49
	mov	#122,r0
L7:
	bra	L49
	mov	#32,r0
L8:
	bra	L49
	mov	#69,r0
L9:
	bra	L49
	mov	#106,r0
L10:
	bra	L49
	mov	#16,r0
L11:
	bra	L49
	mov	#53,r0
L12:
	bra	L49
	mov	#90,r0
L13:
	bra	L49
	mov	#0,r0
L14:
	bra	L49
	mov	#37,r0
L15:
	bra	L49
	mov	#74,r0
L16:
	bra	L49
	mov	#111,r0
L17:
	bra	L49
	mov	#21,r0
L18:
	bra	L49
	mov	#58,r0
L19:
	bra	L49
	mov	#95,r0
L20:
	bra	L49
	mov	#5,r0
L21:
	bra	L49
	mov	#42,r0
L22:
	bra	L49
	mov	#79,r0
L23:
	bra	L49
	mov	#116,r0
L24:
	bra	L49
	mov	#26,r0
L25:
	bra	L49
	mov	#63,r0
L26:
	bra	L49
	mov	#100,r0
L27:
	bra	L49
	mov	#10,r0
L28:
	bra	L49
	mov	#47,r0
L29:
	bra	L49
	mov	#84,r0
L30:
	bra	L49
	mov	#121,r0
L31:
	bra	L49
	mov	#31,r0
L32:
	bra	L49
	mov	#68,r0
L33:
	bra	L49
	mov	#105,r0
L34:
	bra	L49
	mov	#15,r0
L35:
	bra	L49
	mov	#52,r0
L36:
	bra	L49
	mov	#89,r0
L37:
	bra	L49
	mov	#126,r0
L38:
	bra	L49
	mov	#36,r0
L39:
	bra	L49
	mov	#73,r0
L40:
	bra	L49
	mov	#110,r0
L41:
	bra	L49
	mov	#20,r0
L42:
	bra	L49
	mov	#57,r0
L43:
	bra	L49
	mov	#94,r0
L44:
	bra	L49
	mov	#4,r0
L45:
	bra	L49
	mov	#41,r0
L46:
	bra	L49
	mov	#78,r0
L47:
	add	#1,r0
	jsr	@r8
	mov	r0,r4
L49:
	mov	r14,r15
	lds.l	@r15+,pr
	mov.l	@r15+,r14
	rts
	mov.l	@r15+,r8
L51:
	.align 2
L50:
	.long	_sink
