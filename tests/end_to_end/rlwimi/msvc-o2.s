.section .text
test:
/* 00000000 0000  8B 44 24 04 */	mov eax, dword ptr [esp + 4]
/* 00000004 0004  8B 15 00 00 00 00 */	mov edx, dword ptr [_x]
/* 0000000A 000A  83 E2 FE */	and edx, 0xfffffffe
/* 0000000D 000D  8A 08 */	mov cl, byte ptr [eax]
/* 0000000F 000F  83 E1 01 */	and ecx, 1
/* 00000012 0012  0B CA */	or ecx, edx
/* 00000014 0014  89 0D 00 00 00 00 */	mov dword ptr [_x], ecx
/* 0000001A 001A  8A 50 01 */	mov dl, byte ptr [eax + 1]
/* 0000001D 001D  83 E2 01 */	and edx, 1
/* 00000020 0020  83 E1 FD */	and ecx, 0xfffffffd
/* 00000023 0023  D1 E2 */	shl edx, 1
/* 00000025 0025  0B CA */	or ecx, edx
/* 00000027 0027  89 0D 00 00 00 00 */	mov dword ptr [_x], ecx
/* 0000002D 002D  8A 50 02 */	mov dl, byte ptr [eax + 2]
/* 00000030 0030  83 E2 1F */	and edx, 0x1f
/* 00000033 0033  83 E1 83 */	and ecx, 0xffffff83
/* 00000036 0036  C1 E2 02 */	shl edx, 2
/* 00000039 0039  0B CA */	or ecx, edx
/* 0000003B 003B  89 0D 00 00 00 00 */	mov dword ptr [_x], ecx
/* 00000041 0041  8A 40 03 */	mov al, byte ptr [eax + 3]
/* 00000044 0044  83 E0 1F */	and eax, 0x1f
/* 00000047 0047  81 E1 7F F0 FF FF */	and ecx, 0xfffff07f
/* 0000004D 004D  C1 E0 07 */	shl eax, 7
/* 00000050 0050  0B C1 */	or eax, ecx
/* 00000052 0052  8B 0D 00 00 00 00 */	mov ecx, dword ptr [_y]
/* 00000058 0058  83 E1 FD */	and ecx, 0xfffffffd
/* 0000005B 005B  A3 00 00 00 00 */	mov dword ptr [_x], eax
/* 00000060 0060  83 C9 01 */	or ecx, 1
/* 00000063 0063  89 0D 00 00 00 00 */	mov dword ptr [_y], ecx
/* 00000069 0069  C3 */	ret
/* 0000006A 006A  90 */	nop
/* 0000006B 006B  90 */	nop
/* 0000006C 006C  90 */	nop
/* 0000006D 006D  90 */	nop
/* 0000006E 006E  90 */	nop
/* 0000006F 006F  90 */	nop

