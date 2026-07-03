.section .text
test:
/* 00000000 0000  8B 4C 24 04 */	mov ecx, dword ptr [esp + 4]
/* 00000004 0004  8D 41 32 */	lea eax, [ecx + 0x32]
/* 00000007 0007  3D FA 00 00 00 */	cmp eax, 0xfa
/* 0000000C 000C  77 42 */	ja $L105
/* 0000000E 000E  33 D2 */	xor edx, edx
/* 00000010 0010  8A 90 00 00 00 00 */	mov dl, byte ptr [eax + $L110]
/* 00000016 0016  FF 24 95 00 00 00 00 */	jmp dword ptr [edx*4 + $L111]
$L95:
/* 0000001D 001D  8B C1 */	mov eax, ecx
/* 0000001F 001F  0F AF C1 */	imul eax, ecx
/* 00000022 0022  C3 */	ret
$L96:
/* 00000023 0023  49 */	dec ecx
$L97:
/* 00000024 0024  8D 41 01 */	lea eax, [ecx + 1]
/* 00000027 0027  33 C1 */	xor eax, ecx
/* 00000029 0029  C3 */	ret
$L99:
/* 0000002A 002A  49 */	dec ecx
/* 0000002B 002B  B8 02 00 00 00 */	mov eax, 2
/* 00000030 0030  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 00000036 0036  C3 */	ret
$L98:
/* 00000037 0037  41 */	inc ecx
/* 00000038 0038  B8 02 00 00 00 */	mov eax, 2
/* 0000003D 003D  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 00000043 0043  C3 */	ret
$L101:
/* 00000044 0044  03 C9 */	add ecx, ecx
$L102:
/* 00000046 0046  A1 00 00 00 00 */	mov eax, dword ptr [_glob]
/* 0000004B 004B  85 C0 */	test eax, eax
/* 0000004D 004D  75 0A */	jne .L00000059
$L103:
/* 0000004F 004F  49 */	dec ecx
$L105:
/* 00000050 0050  8B C1 */	mov eax, ecx
/* 00000052 0052  99 */	cdq
/* 00000053 0053  2B C2 */	sub eax, edx
/* 00000055 0055  D1 F8 */	sar eax, 1
/* 00000057 0057  8B C8 */	mov ecx, eax
.L00000059:
/* 00000059 0059  89 0D 00 00 00 00 */	mov dword ptr [_glob], ecx
/* 0000005F 005F  B8 02 00 00 00 */	mov eax, 2
/* 00000064 0064  C3 */	ret
/* 00000065 0065  8D 49 00 */	lea ecx, [ecx]
# The tables below are hand-fixed: the COFF disassembler decodes these
# relocated .text dwords (zero-filled before relocation) and the byte-sized
# case-mapping table as code. The .long entries come from the object's dir32
# relocations; the .byte values are the raw section bytes.
$L111:
	.long $L99
	.long $L95
	.long $L96
	.long $L97
	.long $L101
	.long $L98
	.long $L102
	.long $L103
	.long $L105
$L110:
	.byte 0x00
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x01
	.byte 0x02
	.byte 0x03
	.byte 0x08
	.byte 0x08
	.byte 0x04
	.byte 0x04
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x05
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x03
	.byte 0x06
	.byte 0x07
	.byte 0x07
	.byte 0x07
	.byte 0x07
	.byte 0x05
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x08
	.byte 0x03
