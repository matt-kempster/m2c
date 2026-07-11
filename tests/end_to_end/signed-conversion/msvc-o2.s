.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  0F BE C1 */	movsx eax, cl
/* 00000007 0007  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 0000000C 000C  8A D1 */	mov dl, cl
/* 0000000E 000E  D0 E2 */	shl dl, 1
/* 00000010 0010  0F BE C2 */	movsx eax, dl
/* 00000013 0013  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 00000018 0018  8A C1 */	mov al, cl
/* 0000001A 001A  B2 03 */	mov dl, 3
/* 0000001C 001C  F6 EA */	imul dl
/* 0000001E 001E  0F BE C0 */	movsx eax, al
/* 00000021 0021  A3 00 00 00 00 */	mov dword ptr [_glob], eax
/* 00000026 0026  0F BF D1 */	movsx edx, cx
/* 00000029 0029  89 15 00 00 00 00 */	mov dword ptr [_glob], edx
/* 0000002F 002F  8D 04 09 */	lea eax, [ecx + ecx]
/* 00000032 0032  0F BF D0 */	movsx edx, ax
/* 00000035 0035  89 15 00 00 00 00 */	mov dword ptr [_glob], edx
/* 0000003B 003B  8D 04 49 */	lea eax, [ecx + ecx*2]
/* 0000003E 003E  0F BF C8 */	movsx ecx, ax
/* 00000041 0041  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 00000047 0047  C3 */	ret
/* 00000048 0048  90 */	nop
/* 00000049 0049  90 */	nop
/* 0000004A 004A  90 */	nop
/* 0000004B 004B  90 */	nop
/* 0000004C 004C  90 */	nop
/* 0000004D 004D  90 */	nop
/* 0000004E 004E  90 */	nop
/* 0000004F 004F  90 */	nop
