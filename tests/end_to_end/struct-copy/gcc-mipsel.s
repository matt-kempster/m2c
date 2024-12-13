glabel test
   lui     v0,%hi(a)
   addiu   v1,v0,%lo(a)
   lui     v0,%hi(b)
   addiu   v0,v0,%lo(b)
   addiu   a2,v0,0x190
.l14:
   lw      a3,0(v0)
   lw      t0,4(v0)
   lw      t1,8(v0)
   lw      t2,0xc(v0)
   sw      a3,0(v1)
   sw      t0,4(v1)
   sw      t1,8(v1)
   sw      t2,0xc(v1)
   addiu   v0,v0,0x10
   bne     v0,a2,.l14
    addiu   v1,v1,0x10
   or      v0,a1,a0
   andi    v0,v0,0x3
   beqz    v0,.la4
    addiu   v0,a1,0x60
.l50:
   lwl     a3,3(a1)
   lwr     a3,0(a1)
   lwl     t0,7(a1)
   lwr     t0,4(a1)
   lwl     t1,0xb(a1)
   lwr     t1,8(a1)
   lwl     t2,0xf(a1)
   lwr     t2,0xc(a1)
   swl     a3,3(a0)
   swr     a3,0(a0)
   swl     t0,7(a0)
   swr     t0,4(a0)
   swl     t1,0xb(a0)
   swr     t1,8(a0)
   swl     t2,0xf(a0)
   swr     t2,0xc(a0)
   addiu   a1,a1,0x10
   bne     a1,v0,.l50
    addiu   a0,a0,0x10
   j       .ld0
    nop
.la4:
   lw      a3,0(a1)
   lw      t0,4(a1)
   lw      t1,8(a1)
   lw      t2,0xc(a1)
   sw      a3,0(a0)
   sw      t0,4(a0)
   sw      t1,8(a0)
   sw      t2,0xc(a0)
   addiu   a1,a1,0x10
   bne     a1,v0,.la4
    addiu   a0,a0,0x10
.ld0:
   lwl     a3,3(a1)
   lwr     a3,0(a1)
   nop
   swl     a3,3(a0)
   jr      ra
    swr     a3,0(a0)
