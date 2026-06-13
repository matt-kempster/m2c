.include "macros.inc"

.section .text  # 0x0 - 0x68

.global test
test:
subf    r0,r3,r4
cntlzw  r0,r0
srwi    r0,r0,5
stw     r0,global@sda21(r2)
subf    r6,r3,r5
addic   r0,r6,-1
subfe   r0,r0,r6
stw     r0,global@sda21(r2)
xor     r0,r4,r3
srawi   r6,r0,1
and     r0,r0,r4
subf    r0,r0,r6
srwi    r0,r0,0x1f
stw     r0,global@sda21(r2)
srawi   r7,r4,0x1f
srwi    r6,r3,0x1f
subfc   r0,r3,r4
adde    r0,r7,r6
stw     r0,global@sda21(r2)
cntlzw  r0,r3
srwi    r0,r0,5
stw     r0,global@sda21(r2)
addic   r0,r4,-1
subfe   r0,r0,r4
stw     r0,global@sda21(r2)
blr     

.section .sbss  # 0x0 - 0x4

.global global
global:
	.word 0x00000000

