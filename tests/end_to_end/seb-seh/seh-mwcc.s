glabel test
addiu   sp,sp,-0x20
sw      ra,0xc(sp)
sw      s0,8(sp)
sw      a0,0x10(sp)
lw      v0,0x10(sp)
seh     a0,v0
jal     foo
nop
seh     v0,v0
addiu   v0,v0,1
seh     v0,v0
seh     s0,v0
move    v0,s0
lw      ra,0xc(sp)
lw      s0,8(sp)
addiu   sp,sp,0x20
jr      ra
nop
