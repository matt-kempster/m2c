.include "macros.inc"

.section .text

# Test MSVC-mangled C++ symbols with PPC relocation suffixes.
# MSVC symbols start with ? and contain @ characters that must be
# distinguished from @ha/@l/@h/@sda2/@sda21 relocation markers.

.global test
test:
/* 00000000 00000000  3D 60 00 00 */	lis r11, ?TheDebug@@3VDebug@@A@ha
/* 00000004 00000004  3D 40 00 00 */	lis r10, normalSymbol@ha
/* 00000008 00000008  39 6B 00 00 */	addi r11, r11, ?TheDebug@@3VDebug@@A@l
/* 0000000C 0000000C  39 4A 00 00 */	addi r10, r10, normalSymbol@l
/* 00000010 00000010  80 6B 00 00 */	lwz r3, 0(r11)
/* 00000014 00000014  80 0A 00 00 */	lwz r0, 0(r10)
/* 00000018 00000018  7C 63 02 14 */	add r3, r3, r0
/* 0000001C 0000001C  4E 80 00 20 */	blr

.section .bss

.global ?TheDebug@@3VDebug@@A
?TheDebug@@3VDebug@@A:
	.word 0x00000000

.global normalSymbol
normalSymbol:
	.word 0x00000000
