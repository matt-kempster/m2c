.section .text
foo:
/* 00000000 0000  C3 */	ret
/* 00000001 0001  90 */	nop
/* 00000002 0002  90 */	nop
/* 00000003 0003  90 */	nop
/* 00000004 0004  90 */	nop
/* 00000005 0005  90 */	nop
/* 00000006 0006  90 */	nop
/* 00000007 0007  90 */	nop
/* 00000008 0008  90 */	nop
/* 00000009 0009  90 */	nop
/* 0000000A 000A  90 */	nop
/* 0000000B 000B  90 */	nop
/* 0000000C 000C  90 */	nop
/* 0000000D 000D  90 */	nop
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop

test:
/* 00000010 0000  83 EC 08 */	sub esp, 8
/* 00000013 0003  A1 00 00 00 00 */	mov eax, dword ptr ["??_C@_06CBKI@abcdef?$AA@"]
/* 00000018 0008  66 8B 0D 04 00 00 00 */	mov cx, word ptr ["??_C@_06CBKI@abcdef?$AA@" + 0x4]
/* 0000001F 000F  8A 15 06 00 00 00 */	mov dl, byte ptr ["??_C@_06CBKI@abcdef?$AA@" + 0x6]
/* 00000025 0015  89 44 24 00 */	mov dword ptr [esp], eax
/* 00000029 0019  8D 44 24 00 */	lea eax, [esp]
/* 0000002D 001D  66 89 4C 24 04 */	mov word ptr [esp + 4], cx
/* 00000032 0022  50 */	push eax
/* 00000033 0023  88 54 24 0A */	mov byte ptr [esp + 0xa], dl
/* 00000037 0027  E8 00 00 00 00 */	call foo
/* 0000003C 002C  8B 0D 01 00 00 00 */	mov ecx, dword ptr [_a2 + 0x1]
/* 00000042 0032  89 0D 01 00 00 00 */	mov dword ptr [_a1 + 0x1], ecx
/* 00000048 0038  8B 15 00 00 00 00 */	mov edx, dword ptr [_a1]
/* 0000004E 003E  A0 04 00 00 00 */	mov al, byte ptr [_a1 + 0x4]
/* 00000053 0043  8B 0D 00 00 00 00 */	mov ecx, dword ptr ["??_C@_03LKLC@ghi?$AA@"]
/* 00000059 0049  89 15 00 00 00 00 */	mov dword ptr [_a3], edx
/* 0000005F 004F  A2 04 00 00 00 */	mov byte ptr [_a3 + 0x4], al
/* 00000064 0054  89 0D 00 00 00 00 */	mov dword ptr [_buf], ecx
/* 0000006A 005A  83 C4 0C */	add esp, 0xc
/* 0000006D 005D  C3 */	ret
/* 0000006E 005E  90 */	nop
/* 0000006F 005F  90 */	nop

.section .data
"??_C@_03LKLC@ghi?$AA@":
	.long 0x00696867

"??_C@_06CBKI@abcdef?$AA@":
	.long 0x64636261
	.byte 0x65
	.byte 0x66
	.byte 0x00

