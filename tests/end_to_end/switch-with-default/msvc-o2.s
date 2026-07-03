.section .text
test:
/* 00000000 0000  56 */	push esi
/* 00000001 0001  8B 74 24 08 */	mov esi, dword ptr [esp + 8]
/* 00000005 0005  8D 46 FE */	lea eax, [esi - 2]
/* 00000008 0008  83 F8 0B */	cmp eax, 0xb
/* 0000000B 000B  77 1C */	ja $L99
/* 0000000D 000D  33 C9 */	xor ecx, ecx
/* 0000000F 000F  8A 88 00 00 00 00 */	mov cl, byte ptr [eax + $L107]
/* 00000015 0015  FF 24 8D 00 00 00 00 */	jmp dword ptr [ecx*4 + $L108]
$L95:
/* 0000001C 001C  4E */	dec esi
$L96:
/* 0000001D 001D  FF 05 00 00 00 00 */	inc dword ptr [_glob]
/* 00000023 0023  EB 0D */	jmp $L92
$L98:
/* 00000025 0025  03 F6 */	add esi, esi
/* 00000027 0027  EB 09 */	jmp $L92
$L99:
/* 00000029 0029  8B C6 */	mov eax, esi
/* 0000002B 002B  99 */	cdq
/* 0000002C 002C  2B C2 */	sub eax, edx
/* 0000002E 002E  D1 F8 */	sar eax, 1
/* 00000030 0030  8B F0 */	mov esi, eax
$L92:
/* 00000032 0032  8B 15 00 00 00 00 */	mov edx, dword ptr [_glob]
/* 00000038 0038  52 */	push edx
/* 00000039 0039  E8 00 00 00 00 */	call test
/* 0000003E 003E  A1 00 00 00 00 */	mov eax, dword ptr [_glob]
/* 00000043 0043  83 C4 04 */	add esp, 4
/* 00000046 0046  85 C0 */	test eax, eax
/* 00000048 0048  75 06 */	jne .L00000050
/* 0000004A 004A  89 35 00 00 00 00 */	mov dword ptr [_glob], esi
.L00000050:
/* 00000050 0050  B8 02 00 00 00 */	mov eax, 2
/* 00000055 0055  5E */	pop esi
/* 00000056 0056  C3 */	ret
/* 00000057 0057  90 */	nop
# The jump table(s) below are hand-fixed: the COFF disassembler decodes these
# relocated .text dwords (zero-filled before relocation) as code. The entries
# come from the object file's dir32 relocations.
# $L107 is a byte-sized case-mapping table (values transcribed from the raw
# bytes); MSVC maps `x - 2` through it to a jump-table index.
$L108:
	.long $L95
	.long $L96
	.long $L92
	.long $L98
	.long $L99
$L107:
	.byte 0x00
	.byte 0x01
	.byte 0x04
	.byte 0x04
	.byte 0x04
	.byte 0x04
	.byte 0x04
	.byte 0x02
	.byte 0x04
	.byte 0x04
	.byte 0x04
	.byte 0x03
