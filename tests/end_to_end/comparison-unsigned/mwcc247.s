.include "macros.inc"

.section .text

.global test
test:
subfc   r0,r4,r3
subfe   r0,r0,r0
neg     r0,r0
stw     r0,global@sda21(0)
li      r5,-1
subfc   r0,r3,r4
subfze  r0,r5
stw     r0,global@sda21(0)
blr     

.section .sbss  # 0x0 - 0x4

.global global
global:
	.word 0x00000000

