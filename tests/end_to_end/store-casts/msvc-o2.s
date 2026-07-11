.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  8B 4C 24 08 */	mov ecx, dword ptr [esp + 8]
/* 00000008 0008  8B 54 24 0C */	mov edx, dword ptr [esp + 0xc]
/* 0000000C 000C  66 89 48 0A */	mov word ptr [eax + 0xa], cx
/* 00000010 0010  03 CA */	add ecx, edx
/* 00000012 0012  66 C7 40 08 02 00 */	mov word ptr [eax + 8], 2
/* 00000018 0018  66 89 50 0C */	mov word ptr [eax + 0xc], dx
/* 0000001C 001C  66 89 48 0E */	mov word ptr [eax + 0xe], cx
/* 00000020 0020  C3 */	ret
/* 00000021 0021  90 */	nop
/* 00000022 0022  90 */	nop
/* 00000023 0023  90 */	nop
/* 00000024 0024  90 */	nop
/* 00000025 0025  90 */	nop
/* 00000026 0026  90 */	nop
/* 00000027 0027  90 */	nop
/* 00000028 0028  90 */	nop
/* 00000029 0029  90 */	nop
/* 0000002A 002A  90 */	nop
/* 0000002B 002B  90 */	nop
/* 0000002C 002C  90 */	nop
/* 0000002D 002D  90 */	nop
/* 0000002E 002E  90 */	nop
/* 0000002F 002F  90 */	nop
