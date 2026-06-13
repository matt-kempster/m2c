.include "macros.inc"

.section .text  # 0x0 - 0x70

.global test
test:
subf    r6,r3,r4
xor     r0,r4,r3
cntlzw  r7,r6
cntlzw  r6,r3
srwi    r8,r7,5
subf    r7,r3,r5
subf    r5,r5,r3
stw     r8,global@sda21(r2)
or      r7,r7,r5
srawi   r5,r0,1
srwi    r8,r7,0x1f
and     r0,r0,r4
subf    r0,r0,r5
srawi   r7,r4,0x1f
stw     r8,global@sda21(r2)
srwi    r8,r0,0x1f
srwi    r5,r3,0x1f
subfc   r0,r3,r4
stw     r8,global@sda21(r2)
adde    r5,r7,r5
neg     r0,r4
srwi    r3,r6,5
stw     r5,global@sda21(r2)
or      r0,r0,r4
srwi    r0,r0,0x1f
stw     r3,global@sda21(r2)
stw     r0,global@sda21(r2)
blr

.section .sbss  # 0x0 - 0x4

.global global
global:
	.word 0x00000000

