.section .text
foo:
/* 00000000 0000  8B 44 24 08 */	mov eax, dword ptr [esp + 8]
/* 00000004 0004  8B 54 24 04 */	mov edx, dword ptr [esp + 4]
/* 00000008 0008  8B 08 */	mov ecx, dword ptr [eax]
/* 0000000A 000A  89 0A */	mov dword ptr [edx], ecx
/* 0000000C 000C  C3 */	ret
/* 0000000D 000D  90 */	nop
/* 0000000E 000E  90 */	nop
/* 0000000F 000F  90 */	nop

test:
/* 00000010 0000  8D 44 24 04 */	lea eax, [esp + 4]
/* 00000014 0004  8D 54 24 08 */	lea edx, [esp + 8]
/* 00000018 0008  89 44 24 04 */	mov dword ptr [esp + 4], eax
/* 0000001C 000C  8D 44 24 04 */	lea eax, [esp + 4]
/* 00000020 0010  8D 4C 24 08 */	lea ecx, [esp + 8]
/* 00000024 0014  52 */	push edx
/* 00000025 0015  50 */	push eax
/* 00000026 0016  89 4C 24 10 */	mov dword ptr [esp + 0x10], ecx
/* 0000002A 001A  E8 00 00 00 00 */	call foo
/* 0000002F 001F  8B 44 24 10 */	mov eax, dword ptr [esp + 0x10]
/* 00000033 0023  50 */	push eax
/* 00000034 0024  50 */	push eax
/* 00000035 0025  89 44 24 14 */	mov dword ptr [esp + 0x14], eax
/* 00000039 0029  E8 00 00 00 00 */	call foo
/* 0000003E 002E  83 C4 10 */	add esp, 0x10
/* 00000041 0031  C3 */	ret
/* 00000042 0032  90 */	nop
/* 00000043 0033  90 */	nop
/* 00000044 0034  90 */	nop
/* 00000045 0035  90 */	nop
/* 00000046 0036  90 */	nop
/* 00000047 0037  90 */	nop
/* 00000048 0038  90 */	nop
/* 00000049 0039  90 */	nop
/* 0000004A 003A  90 */	nop
/* 0000004B 003B  90 */	nop
/* 0000004C 003C  90 */	nop
/* 0000004D 003D  90 */	nop
/* 0000004E 003E  90 */	nop
/* 0000004F 003F  90 */	nop

