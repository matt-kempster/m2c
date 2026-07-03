.section .text
test:
/* 00000000 0000  83 EC 08 */	sub esp, 8
/* 00000003 0003  66 8B 0D 04 00 00 00 */	mov cx, word ptr ["??_C@_05DLON@hello?$AA@" + 0x4]
/* 0000000A 000A  A1 00 00 00 00 */	mov eax, dword ptr ["??_C@_05DLON@hello?$AA@"]
/* 0000000F 000F  66 89 4C 24 04 */	mov word ptr [esp + 4], cx
/* 00000014 0014  8B 4C 24 0C */	mov ecx, dword ptr [esp + 0xc]
/* 00000018 0018  89 44 24 00 */	mov dword ptr [esp], eax
/* 0000001C 001C  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 00000020 0020  0F BE 54 0C 00 */	movsx edx, byte ptr [esp + ecx]
/* 00000025 0025  8B 04 88 */	mov eax, dword ptr [eax + ecx*4]
/* 00000028 0028  56 */	push esi
/* 00000029 0029  0F AF C2 */	imul eax, edx
/* 0000002C 002C  8B 54 24 18 */	mov edx, dword ptr [esp + 0x18]
/* 00000030 0030  8B 74 8A 04 */	mov esi, dword ptr [edx + ecx*4 + 4]
/* 00000034 0034  0F BF 0C 4D 00 00 00 00 */	movsx ecx, word ptr [ecx*2 + _globalArray]
/* 0000003C 003C  03 C6 */	add eax, esi
/* 0000003E 003E  5E */	pop esi
/* 0000003F 003F  03 C1 */	add eax, ecx
/* 00000041 0041  83 C4 08 */	add esp, 8
/* 00000044 0044  C3 */	ret
/* 00000045 0045  90 */	nop
/* 00000046 0046  90 */	nop
/* 00000047 0047  90 */	nop
/* 00000048 0048  90 */	nop
/* 00000049 0049  90 */	nop
/* 0000004A 004A  90 */	nop
/* 0000004B 004B  90 */	nop
/* 0000004C 004C  90 */	nop
/* 0000004D 004D  90 */	nop
/* 0000004E 004E  90 */	nop
/* 0000004F 004F  90 */	nop

.section .data
"??_C@_05DLON@hello?$AA@":
	.long 0x6C6C6568
	.byte 0x6F
	.byte 0x00

