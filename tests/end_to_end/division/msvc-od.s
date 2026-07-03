.section .text
test:
/* 00000000 0000  55 */	push ebp
/* 00000001 0001  8B EC */	mov ebp, esp
/* 00000003 0003  8B 45 08 */	mov eax, dword ptr [ebp + 8]
/* 00000006 0006  99 */	cdq
/* 00000007 0007  F7 7D 0C */	idiv dword ptr [ebp + 0xc]
/* 0000000A 000A  5D */	pop ebp
/* 0000000B 000B  C3 */	ret

