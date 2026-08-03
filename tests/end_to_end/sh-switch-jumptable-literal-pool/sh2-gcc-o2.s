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
	mov.l	L252,r8
	jsr	@r8
	mov	r15,r14
	extu.b	r0,r4
	mov.w	L253,r1
	cmp/hi	r1,r4
	bf.s	LF100
	mov	r4,r1
	bra	L247
	nop
LF100:
	add	r1,r1
	mova	L248,r0
	mov.w	@(r0,r1),r1
	add	r1,r0
	bra	L251
	nop
	.align 1
L253:
	.short	243
L254:
	.align 2
L252:
	.long	_sink
L251:
	jmp        @r0
	nop
	.align 2
L248:
	.word	L3-L248
	.word	L4-L248
	.word	L5-L248
	.word	L6-L248
	.word	L7-L248
	.word	L8-L248
	.word	L9-L248
	.word	L10-L248
	.word	L11-L248
	.word	L12-L248
	.word	L13-L248
	.word	L14-L248
	.word	L15-L248
	.word	L16-L248
	.word	L17-L248
	.word	L18-L248
	.word	L19-L248
	.word	L20-L248
	.word	L21-L248
	.word	L22-L248
	.word	L23-L248
	.word	L24-L248
	.word	L25-L248
	.word	L26-L248
	.word	L27-L248
	.word	L28-L248
	.word	L29-L248
	.word	L30-L248
	.word	L31-L248
	.word	L32-L248
	.word	L33-L248
	.word	L34-L248
	.word	L35-L248
	.word	L36-L248
	.word	L37-L248
	.word	L38-L248
	.word	L39-L248
	.word	L40-L248
	.word	L41-L248
	.word	L42-L248
	.word	L43-L248
	.word	L44-L248
	.word	L45-L248
	.word	L46-L248
	.word	L47-L248
	.word	L48-L248
	.word	L49-L248
	.word	L50-L248
	.word	L51-L248
	.word	L52-L248
	.word	L53-L248
	.word	L54-L248
	.word	L55-L248
	.word	L56-L248
	.word	L57-L248
	.word	L58-L248
	.word	L59-L248
	.word	L60-L248
	.word	L61-L248
	.word	L62-L248
	.word	L63-L248
	.word	L64-L248
	.word	L65-L248
	.word	L66-L248
	.word	L67-L248
	.word	L68-L248
	.word	L69-L248
	.word	L70-L248
	.word	L71-L248
	.word	L72-L248
	.word	L73-L248
	.word	L74-L248
	.word	L75-L248
	.word	L76-L248
	.word	L77-L248
	.word	L78-L248
	.word	L79-L248
	.word	L80-L248
	.word	L81-L248
	.word	L82-L248
	.word	L83-L248
	.word	L84-L248
	.word	L85-L248
	.word	L86-L248
	.word	L87-L248
	.word	L88-L248
	.word	L89-L248
	.word	L90-L248
	.word	L91-L248
	.word	L92-L248
	.word	L93-L248
	.word	L94-L248
	.word	L95-L248
	.word	L96-L248
	.word	L97-L248
	.word	L98-L248
	.word	L99-L248
	.word	L100-L248
	.word	L101-L248
	.word	L102-L248
	.word	L103-L248
	.word	L104-L248
	.word	L105-L248
	.word	L106-L248
	.word	L107-L248
	.word	L108-L248
	.word	L109-L248
	.word	L110-L248
	.word	L111-L248
	.word	L112-L248
	.word	L113-L248
	.word	L114-L248
	.word	L115-L248
	.word	L116-L248
	.word	L117-L248
	.word	L118-L248
	.word	L119-L248
	.word	L120-L248
	.word	L121-L248
	.word	L122-L248
	.word	L123-L248
	.word	L124-L248
	.word	L125-L248
	.word	L126-L248
	.word	L127-L248
	.word	L128-L248
	.word	L129-L248
	.word	L130-L248
	.word	L131-L248
	.word	L132-L248
	.word	L133-L248
	.word	L134-L248
	.word	L135-L248
	.word	L136-L248
	.word	L137-L248
	.word	L138-L248
	.word	L139-L248
	.word	L140-L248
	.word	L141-L248
	.word	L142-L248
	.word	L143-L248
	.word	L144-L248
	.word	L145-L248
	.word	L146-L248
	.word	L147-L248
	.word	L148-L248
	.word	L149-L248
	.word	L150-L248
	.word	L151-L248
	.word	L152-L248
	.word	L153-L248
	.word	L154-L248
	.word	L155-L248
	.word	L156-L248
	.word	L157-L248
	.word	L158-L248
	.word	L159-L248
	.word	L160-L248
	.word	L161-L248
	.word	L162-L248
	.word	L163-L248
	.word	L164-L248
	.word	L165-L248
	.word	L166-L248
	.word	L167-L248
	.word	L168-L248
	.word	L169-L248
	.word	L170-L248
	.word	L171-L248
	.word	L172-L248
	.word	L173-L248
	.word	L174-L248
	.word	L175-L248
	.word	L176-L248
	.word	L177-L248
	.word	L178-L248
	.word	L179-L248
	.word	L180-L248
	.word	L181-L248
	.word	L182-L248
	.word	L183-L248
	.word	L184-L248
	.word	L185-L248
	.word	L186-L248
	.word	L187-L248
	.word	L188-L248
	.word	L189-L248
	.word	L190-L248
	.word	L191-L248
	.word	L192-L248
	.word	L193-L248
	.word	L194-L248
	.word	L195-L248
	.word	L196-L248
	.word	L197-L248
	.word	L198-L248
	.word	L199-L248
	.word	L200-L248
	.word	L201-L248
	.word	L202-L248
	.word	L203-L248
	.word	L204-L248
	.word	L205-L248
	.word	L206-L248
	.word	L207-L248
	.word	L208-L248
	.word	L209-L248
	.word	L210-L248
	.word	L211-L248
	.word	L212-L248
	.word	L213-L248
	.word	L214-L248
	.word	L215-L248
	.word	L216-L248
	.word	L217-L248
	.word	L218-L248
	.word	L219-L248
	.word	L220-L248
	.word	L221-L248
	.word	L222-L248
	.word	L223-L248
	.word	L224-L248
	.word	L225-L248
	.word	L226-L248
	.word	L227-L248
	.word	L228-L248
	.word	L229-L248
	.word	L230-L248
	.word	L231-L248
	.word	L232-L248
	.word	L233-L248
	.word	L234-L248
	.word	L235-L248
	.word	L236-L248
	.word	L237-L248
	.word	L238-L248
	.word	L239-L248
	.word	L240-L248
	.word	L241-L248
	.word	L242-L248
	.word	L243-L248
	.word	L244-L248
	.word	L245-L248
	.word	L246-L248
L3:
	mov.w	L255,r1
	bra	L250
	nop
L4:
	mov.w	L256,r1
	bra	L250
	nop
L5:
	mov.w	L257,r1
	bra	L250
	nop
L6:
	mov.w	L258,r1
	bra	L250
	nop
L7:
	mov.w	L259,r1
	bra	L250
	nop
L8:
	mov.w	L260,r1
	bra	L250
	nop
L9:
	mov.w	L261,r1
	bra	L250
	nop
L10:
	mov.w	L262,r1
	bra	L250
	nop
L11:
	mov.w	L263,r1
	bra	L250
	nop
L12:
	mov.w	L264,r1
	bra	L250
	nop
L13:
	mov.w	L265,r1
	bra	L250
	nop
L14:
	mov.w	L266,r1
	bra	L250
	nop
L15:
	mov.w	L267,r1
	bra	L250
	nop
L16:
	mov.w	L268,r1
	bra	L250
	nop
L17:
	mov.w	L269,r1
	bra	L250
	nop
L18:
	mov.w	L270,r1
	bra	L250
	nop
L19:
	mov.w	L271,r1
	bra	L250
	nop
L20:
	mov.w	L272,r1
	bra	L250
	nop
L21:
	mov.w	L273,r1
	bra	L250
	nop
L22:
	mov.w	L274,r1
	bra	L250
	nop
L23:
	mov.w	L275,r1
	bra	L250
	nop
L24:
	mov.w	L276,r1
	bra	L250
	nop
L25:
	mov.w	L277,r1
	bra	L250
	nop
L26:
	mov.w	L278,r1
	bra	L250
	nop
L27:
	mov.w	L279,r1
	bra	L250
	nop
L28:
	mov.w	L280,r1
	bra	L250
	nop
L29:
	mov.w	L281,r1
	bra	L250
	nop
L30:
	mov.w	L282,r1
	bra	L250
	nop
	.align 1
L255:
	.short	131
L256:
	.short	388
L257:
	.short	645
L258:
	.short	902
L259:
	.short	1159
L260:
	.short	1416
L261:
	.short	1673
L262:
	.short	1930
L263:
	.short	2187
L264:
	.short	2444
L265:
	.short	2701
L266:
	.short	2958
L267:
	.short	3215
L268:
	.short	3472
L269:
	.short	3729
L270:
	.short	3986
L271:
	.short	4243
L272:
	.short	4500
L273:
	.short	4757
L274:
	.short	5014
L275:
	.short	5271
L276:
	.short	5528
L277:
	.short	5785
L278:
	.short	6042
L279:
	.short	6299
L280:
	.short	6556
L281:
	.short	6813
L282:
	.short	7070
L31:
	mov.w	L283,r1
	bra	L250
	nop
L32:
	mov.w	L284,r1
	bra	L250
	nop
L33:
	mov.w	L285,r1
	bra	L250
	nop
L34:
	mov.w	L286,r1
	bra	L250
	nop
L35:
	mov.w	L287,r1
	bra	L250
	nop
L36:
	mov.w	L288,r1
	bra	L250
	nop
L37:
	mov.w	L289,r1
	bra	L250
	nop
L38:
	mov.w	L290,r1
	bra	L250
	nop
L39:
	mov.w	L291,r1
	bra	L250
	nop
L40:
	mov.w	L292,r1
	bra	L250
	nop
L41:
	mov.w	L293,r1
	bra	L250
	nop
L42:
	mov.w	L294,r1
	bra	L250
	nop
L43:
	mov.w	L295,r1
	bra	L250
	nop
L44:
	mov.w	L296,r1
	bra	L250
	nop
L45:
	mov.w	L297,r1
	bra	L250
	nop
L46:
	mov.w	L298,r1
	bra	L250
	nop
L47:
	mov.w	L299,r1
	bra	L250
	nop
L48:
	mov.w	L300,r1
	bra	L250
	nop
L49:
	mov.w	L301,r1
	bra	L250
	nop
L50:
	mov.w	L302,r1
	bra	L250
	nop
L51:
	mov.w	L303,r1
	bra	L250
	nop
L52:
	mov.w	L304,r1
	bra	L250
	nop
L53:
	mov.w	L305,r1
	bra	L250
	nop
L54:
	mov.w	L306,r1
	bra	L250
	nop
L55:
	mov.w	L307,r1
	bra	L250
	nop
L56:
	mov.w	L308,r1
	bra	L250
	nop
L57:
	mov.w	L309,r1
	bra	L250
	nop
L58:
	mov.w	L310,r1
	bra	L250
	nop
	.align 1
L283:
	.short	7327
L284:
	.short	7584
L285:
	.short	7841
L286:
	.short	8098
L287:
	.short	8355
L288:
	.short	8612
L289:
	.short	8869
L290:
	.short	9126
L291:
	.short	9383
L292:
	.short	9640
L293:
	.short	9897
L294:
	.short	10154
L295:
	.short	10411
L296:
	.short	10668
L297:
	.short	10925
L298:
	.short	11182
L299:
	.short	11439
L300:
	.short	11696
L301:
	.short	11953
L302:
	.short	12210
L303:
	.short	12467
L304:
	.short	12724
L305:
	.short	12981
L306:
	.short	13238
L307:
	.short	13495
L308:
	.short	13752
L309:
	.short	14009
L310:
	.short	14266
L59:
	mov.w	L311,r1
	bra	L250
	nop
L60:
	mov.w	L312,r1
	bra	L250
	nop
L61:
	mov.w	L313,r1
	bra	L250
	nop
L62:
	mov.w	L314,r1
	bra	L250
	nop
L63:
	mov.w	L315,r1
	bra	L250
	nop
L64:
	mov.w	L316,r1
	bra	L250
	nop
L65:
	mov.w	L317,r1
	bra	L250
	nop
L66:
	mov.w	L318,r1
	bra	L250
	nop
L67:
	mov.w	L319,r1
	bra	L250
	nop
L68:
	mov.w	L320,r1
	bra	L250
	nop
L69:
	mov.w	L321,r1
	bra	L250
	nop
L70:
	mov.w	L322,r1
	bra	L250
	nop
L71:
	mov.w	L323,r1
	bra	L250
	nop
L72:
	mov.w	L324,r1
	bra	L250
	nop
L73:
	mov.w	L325,r1
	bra	L250
	nop
L74:
	mov.w	L326,r1
	bra	L250
	nop
L75:
	mov.w	L327,r1
	bra	L250
	nop
L76:
	mov.w	L328,r1
	bra	L250
	nop
L77:
	mov.w	L329,r1
	bra	L250
	nop
L78:
	mov.w	L330,r1
	bra	L250
	nop
L79:
	mov.w	L331,r1
	bra	L250
	nop
L80:
	mov.w	L332,r1
	bra	L250
	nop
L81:
	mov.w	L333,r1
	bra	L250
	nop
L82:
	mov.w	L334,r1
	bra	L250
	nop
L83:
	mov.w	L335,r1
	bra	L250
	nop
L84:
	mov.w	L336,r1
	bra	L250
	nop
L85:
	mov.w	L337,r1
	bra	L250
	nop
L86:
	mov.w	L338,r1
	bra	L250
	nop
	.align 1
L311:
	.short	14523
L312:
	.short	14780
L313:
	.short	15037
L314:
	.short	15294
L315:
	.short	15551
L316:
	.short	15808
L317:
	.short	16065
L318:
	.short	16322
L319:
	.short	16579
L320:
	.short	16836
L321:
	.short	17093
L322:
	.short	17350
L323:
	.short	17607
L324:
	.short	17864
L325:
	.short	18121
L326:
	.short	18378
L327:
	.short	18635
L328:
	.short	18892
L329:
	.short	19149
L330:
	.short	19406
L331:
	.short	19663
L332:
	.short	19920
L333:
	.short	20177
L334:
	.short	20434
L335:
	.short	20691
L336:
	.short	20948
L337:
	.short	21205
L338:
	.short	21462
L87:
	mov.w	L339,r1
	bra	L250
	nop
L88:
	mov.w	L340,r1
	bra	L250
	nop
L89:
	mov.w	L341,r1
	bra	L250
	nop
L90:
	mov.w	L342,r1
	bra	L250
	nop
L91:
	mov.w	L343,r1
	bra	L250
	nop
L92:
	mov.w	L344,r1
	bra	L250
	nop
L93:
	mov.w	L345,r1
	bra	L250
	nop
L94:
	mov.w	L346,r1
	bra	L250
	nop
L95:
	mov.w	L347,r1
	bra	L250
	nop
L96:
	mov.w	L348,r1
	bra	L250
	nop
L97:
	mov.w	L349,r1
	bra	L250
	nop
L98:
	mov.w	L350,r1
	bra	L250
	nop
L99:
	mov.w	L351,r1
	bra	L250
	nop
L100:
	mov.w	L352,r1
	bra	L250
	nop
L101:
	mov.w	L353,r1
	bra	L250
	nop
L102:
	mov.w	L354,r1
	bra	L250
	nop
L103:
	mov.w	L355,r1
	bra	L250
	nop
L104:
	mov.w	L356,r1
	bra	L250
	nop
L105:
	mov.w	L357,r1
	bra	L250
	nop
L106:
	mov.w	L358,r1
	bra	L250
	nop
L107:
	mov.w	L359,r1
	bra	L250
	nop
L108:
	mov.w	L360,r1
	bra	L250
	nop
L109:
	mov.w	L361,r1
	bra	L250
	nop
L110:
	mov.w	L362,r1
	bra	L250
	nop
L111:
	mov.w	L363,r1
	bra	L250
	nop
L112:
	mov.w	L364,r1
	bra	L250
	nop
L113:
	mov.w	L365,r1
	bra	L250
	nop
L114:
	mov.w	L366,r1
	bra	L250
	nop
	.align 1
L339:
	.short	21719
L340:
	.short	21976
L341:
	.short	22233
L342:
	.short	22490
L343:
	.short	22747
L344:
	.short	23004
L345:
	.short	23261
L346:
	.short	23518
L347:
	.short	23775
L348:
	.short	24032
L349:
	.short	24289
L350:
	.short	24546
L351:
	.short	24803
L352:
	.short	25060
L353:
	.short	25317
L354:
	.short	25574
L355:
	.short	25831
L356:
	.short	26088
L357:
	.short	26345
L358:
	.short	26602
L359:
	.short	26859
L360:
	.short	27116
L361:
	.short	27373
L362:
	.short	27630
L363:
	.short	27887
L364:
	.short	28144
L365:
	.short	28401
L366:
	.short	28658
L115:
	mov.w	L367,r1
	bra	L250
	nop
L116:
	mov.w	L368,r1
	bra	L250
	nop
L117:
	mov.w	L369,r1
	bra	L250
	nop
L118:
	mov.w	L370,r1
	bra	L250
	nop
L119:
	mov.w	L371,r1
	bra	L250
	nop
L120:
	mov.w	L372,r1
	bra	L250
	nop
L121:
	mov.w	L373,r1
	bra	L250
	nop
L122:
	mov.w	L374,r1
	bra	L250
	nop
L123:
	mov.w	L375,r1
	bra	L250
	nop
L124:
	mov.w	L376,r1
	bra	L250
	nop
L125:
	mov.w	L377,r1
	bra	L250
	nop
L126:
	mov.w	L378,r1
	bra	L250
	nop
L127:
	mov.w	L379,r1
	bra	L250
	nop
L128:
	mov.w	L380,r1
	bra	L250
	nop
L129:
	mov.w	L381,r1
	bra	L250
	nop
L130:
	mov.l	L382,r1
	bra	L250
	nop
L131:
	mov.l	L383,r1
	bra	L250
	nop
L132:
	mov.l	L384,r1
	bra	L250
	nop
L133:
	mov.l	L385,r1
	bra	L250
	nop
L134:
	mov.l	L386,r1
	bra	L250
	nop
L135:
	mov.l	L387,r1
	bra	L250
	nop
L136:
	mov.l	L388,r1
	bra	L250
	nop
L137:
	mov.l	L389,r1
	bra	L250
	nop
L138:
	mov.l	L390,r1
	bra	L250
	nop
L139:
	mov.l	L391,r1
	bra	L250
	nop
L140:
	mov.l	L392,r1
	bra	L250
	nop
L141:
	mov.l	L393,r1
	bra	L250
	nop
L142:
	mov.l	L394,r1
	bra	L250
	nop
	.align 1
L367:
	.short	28915
L368:
	.short	29172
L369:
	.short	29429
L370:
	.short	29686
L371:
	.short	29943
L372:
	.short	30200
L373:
	.short	30457
L374:
	.short	30714
L375:
	.short	30971
L376:
	.short	31228
L377:
	.short	31485
L378:
	.short	31742
L379:
	.short	31999
L380:
	.short	32256
L381:
	.short	32513
L395:
	.align 2
L382:
	.long	32770
L383:
	.long	33027
L384:
	.long	33284
L385:
	.long	33541
L386:
	.long	33798
L387:
	.long	34055
L388:
	.long	34312
L389:
	.long	34569
L390:
	.long	34826
L391:
	.long	35083
L392:
	.long	35340
L393:
	.long	35597
L394:
	.long	35854
L143:
	mov.l	L396,r1
	bra	L250
	nop
L144:
	mov.l	L397,r1
	bra	L250
	nop
L145:
	mov.l	L398,r1
	bra	L250
	nop
L146:
	mov.l	L399,r1
	bra	L250
	nop
L147:
	mov.l	L400,r1
	bra	L250
	nop
L148:
	mov.l	L401,r1
	bra	L250
	nop
L149:
	mov.l	L402,r1
	bra	L250
	nop
L150:
	mov.l	L403,r1
	bra	L250
	nop
L151:
	mov.l	L404,r1
	bra	L250
	nop
L152:
	mov.l	L405,r1
	bra	L250
	nop
L153:
	mov.l	L406,r1
	bra	L250
	nop
L154:
	mov.l	L407,r1
	bra	L250
	nop
L155:
	mov.l	L408,r1
	bra	L250
	nop
L156:
	mov.l	L409,r1
	bra	L250
	nop
L157:
	mov.l	L410,r1
	bra	L250
	nop
L158:
	mov.l	L411,r1
	bra	L250
	nop
L159:
	mov.l	L412,r1
	bra	L250
	nop
L160:
	mov.l	L413,r1
	bra	L250
	nop
L161:
	mov.l	L414,r1
	bra	L250
	nop
L162:
	mov.l	L415,r1
	bra	L250
	nop
L163:
	mov.l	L416,r1
	bra	L250
	nop
L164:
	mov.l	L417,r1
	bra	L250
	nop
L165:
	mov.l	L418,r1
	bra	L250
	nop
L166:
	mov.l	L419,r1
	bra	L250
	nop
L167:
	mov.l	L420,r1
	bra	L250
	nop
L168:
	mov.l	L421,r1
	bra	L250
	nop
L169:
	mov.l	L422,r1
	bra	L250
	nop
L170:
	mov.l	L423,r1
	bra	L250
	nop
L171:
	mov.l	L424,r1
	bra	L250
	nop
L172:
	mov.l	L425,r1
	bra	L250
	nop
L173:
	mov.l	L426,r1
	bra	L250
	nop
L174:
	mov.l	L427,r1
	bra	L250
	nop
L175:
	mov.l	L428,r1
	bra	L250
	nop
L176:
	mov.l	L429,r1
	bra	L250
	nop
L177:
	mov.l	L430,r1
	bra	L250
	nop
L178:
	mov.l	L431,r1
	bra	L250
	nop
L179:
	mov.l	L432,r1
	bra	L250
	nop
L180:
	mov.l	L433,r1
	bra	L250
	nop
L181:
	mov.l	L434,r1
	bra	L250
	nop
L182:
	mov.l	L435,r1
	bra	L250
	nop
L183:
	mov.l	L436,r1
	bra	L250
	nop
L184:
	mov.l	L437,r1
	bra	L250
	nop
L185:
	mov.l	L438,r1
	bra	L250
	nop
L186:
	mov.l	L439,r1
	bra	L250
	nop
L187:
	mov.l	L440,r1
	bra	L250
	nop
L188:
	mov.l	L441,r1
	bra	L250
	nop
L189:
	mov.l	L442,r1
	bra	L250
	nop
L190:
	mov.l	L443,r1
	bra	L250
	nop
L191:
	mov.l	L444,r1
	bra	L250
	nop
L192:
	mov.l	L445,r1
	bra	L250
	nop
L193:
	mov.l	L446,r1
	bra	L250
	nop
L194:
	mov.l	L447,r1
	bra	L250
	nop
L195:
	mov.l	L448,r1
	bra	L250
	nop
L196:
	mov.l	L449,r1
	bra	L250
	nop
L197:
	mov.l	L450,r1
	bra	L250
	nop
L198:
	mov.l	L451,r1
	bra	L250
	nop
L452:
	.align 2
L396:
	.long	36111
L397:
	.long	36368
L398:
	.long	36625
L399:
	.long	36882
L400:
	.long	37139
L401:
	.long	37396
L402:
	.long	37653
L403:
	.long	37910
L404:
	.long	38167
L405:
	.long	38424
L406:
	.long	38681
L407:
	.long	38938
L408:
	.long	39195
L409:
	.long	39452
L410:
	.long	39709
L411:
	.long	39966
L412:
	.long	40223
L413:
	.long	40480
L414:
	.long	40737
L415:
	.long	40994
L416:
	.long	41251
L417:
	.long	41508
L418:
	.long	41765
L419:
	.long	42022
L420:
	.long	42279
L421:
	.long	42536
L422:
	.long	42793
L423:
	.long	43050
L424:
	.long	43307
L425:
	.long	43564
L426:
	.long	43821
L427:
	.long	44078
L428:
	.long	44335
L429:
	.long	44592
L430:
	.long	44849
L431:
	.long	45106
L432:
	.long	45363
L433:
	.long	45620
L434:
	.long	45877
L435:
	.long	46134
L436:
	.long	46391
L437:
	.long	46648
L438:
	.long	46905
L439:
	.long	47162
L440:
	.long	47419
L441:
	.long	47676
L442:
	.long	47933
L443:
	.long	48190
L444:
	.long	48447
L445:
	.long	48704
L446:
	.long	48961
L447:
	.long	49218
L448:
	.long	49475
L449:
	.long	49732
L450:
	.long	49989
L451:
	.long	50246
L199:
	mov.l	L453,r1
	bra	L250
	nop
L200:
	mov.l	L454,r1
	bra	L250
	nop
L201:
	mov.l	L455,r1
	bra	L250
	nop
L202:
	mov.l	L456,r1
	bra	L250
	nop
L203:
	mov.l	L457,r1
	bra	L250
	nop
L204:
	mov.l	L458,r1
	bra	L250
	nop
L205:
	mov.l	L459,r1
	bra	L250
	nop
L206:
	mov.l	L460,r1
	bra	L250
	nop
L207:
	mov.l	L461,r1
	bra	L250
	nop
L208:
	mov.l	L462,r1
	bra	L250
	nop
L209:
	mov.l	L463,r1
	bra	L250
	nop
L210:
	mov.l	L464,r1
	bra	L250
	nop
L211:
	mov.l	L465,r1
	bra	L250
	nop
L212:
	mov.l	L466,r1
	bra	L250
	nop
L213:
	mov.l	L467,r1
	bra	L250
	nop
L214:
	mov.l	L468,r1
	bra	L250
	nop
L215:
	mov.l	L469,r1
	bra	L250
	nop
L216:
	mov.l	L470,r1
	bra	L250
	nop
L217:
	mov.l	L471,r1
	bra	L250
	nop
L218:
	mov.l	L472,r1
	bra	L250
	nop
L219:
	mov.l	L473,r1
	bra	L250
	nop
L220:
	mov.l	L474,r1
	bra	L250
	nop
L221:
	mov.l	L475,r1
	bra	L250
	nop
L222:
	mov.l	L476,r1
	bra	L250
	nop
L223:
	mov.l	L477,r1
	bra	L250
	nop
L224:
	mov.l	L478,r1
	bra	L250
	nop
L225:
	mov.l	L479,r1
	bra	L250
	nop
L226:
	mov.l	L480,r1
	bra	L250
	nop
L227:
	mov.l	L481,r1
	bra	L250
	nop
L228:
	mov.l	L482,r1
	bra	L250
	nop
L229:
	mov.l	L483,r1
	bra	L250
	nop
L230:
	mov.l	L484,r1
	bra	L250
	nop
L231:
	mov.l	L485,r1
	bra	L250
	nop
L232:
	mov.l	L486,r1
	bra	L250
	nop
L233:
	mov.l	L487,r1
	bra	L250
	nop
L234:
	mov.l	L488,r1
	bra	L250
	nop
L235:
	mov.l	L489,r1
	bra	L250
	nop
L236:
	mov.l	L490,r1
	bra	L250
	nop
L237:
	mov.l	L491,r1
	bra	L250
	nop
L238:
	mov.l	L492,r1
	bra	L250
	nop
L239:
	mov.l	L493,r1
	bra	L250
	nop
L240:
	mov.l	L494,r1
	bra	L250
	nop
L241:
	mov.l	L495,r1
	bra	L250
	nop
L242:
	mov.l	L496,r1
	bra	L250
	nop
L243:
	mov.l	L497,r1
	bra	L250
	nop
L244:
	mov.l	L498,r1
	bra	L250
	nop
L245:
	mov.l	L499,r1
	bra	L250
	nop
L246:
	mov.l	L500,r1
L250:
	mov.l	L501,r0
	jsr	@r0
	add	r1,r4
	bra	L503
	mov	r14,r15
L247:
	jsr	@r8
	add	#-1,r4
	mov	r14,r15
L503:
	lds.l	@r15+,pr
	mov.l	@r15+,r14
	rts
	mov.l	@r15+,r8
L502:
	.align 2
L453:
	.long	50503
L454:
	.long	50760
L455:
	.long	51017
L456:
	.long	51274
L457:
	.long	51531
L458:
	.long	51788
L459:
	.long	52045
L460:
	.long	52302
L461:
	.long	52559
L462:
	.long	52816
L463:
	.long	53073
L464:
	.long	53330
L465:
	.long	53587
L466:
	.long	53844
L467:
	.long	54101
L468:
	.long	54358
L469:
	.long	54615
L470:
	.long	54872
L471:
	.long	55129
L472:
	.long	55386
L473:
	.long	55643
L474:
	.long	55900
L475:
	.long	56157
L476:
	.long	56414
L477:
	.long	56671
L478:
	.long	56928
L479:
	.long	57185
L480:
	.long	57442
L481:
	.long	57699
L482:
	.long	57956
L483:
	.long	58213
L484:
	.long	58470
L485:
	.long	58727
L486:
	.long	58984
L487:
	.long	59241
L488:
	.long	59498
L489:
	.long	59755
L490:
	.long	60012
L491:
	.long	60269
L492:
	.long	60526
L493:
	.long	60783
L494:
	.long	61040
L495:
	.long	61297
L496:
	.long	61554
L497:
	.long	61811
L498:
	.long	62068
L499:
	.long	62325
L500:
	.long	62582
L501:
	.long	_sink
