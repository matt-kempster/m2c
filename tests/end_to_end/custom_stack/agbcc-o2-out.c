struct _m2c_stack_test {
    /* 0x00 */ char pad0[4];
    /* 0x04 */ s32 sp4;                             /* inferred */
    /* 0x08 */ s32 sp8;                             /* inferred */
    /* 0x0C */ s32 spC;                             /* inferred */
    /* 0x10 */ s32 sp10;                            /* inferred */
    /* 0x14 */ s32 sp14;                            /* inferred */
    /* 0x18 */ char pad18[0x14];
};                                                  /* size = 0x2C */

? frob(s32 *);                                      /* static */

s32 test(void *arg0) {
    s32 sp4;
    s32 sp8;
    s32 spC;
    s32 sp10;
    s32 sp14;                                       /* compiler-managed */
    s32 temp_r1;
    s32 temp_r2;
    s32 temp_r3;
    s32 temp_r5;

    frob(&spC);
    frob(&subroutine_arg0 + 0xE);
    frob(&sp10);
    frob(&sp14);
    frob(&subroutine_arg0);
    temp_r1 = arg0->unk0;
    temp_r3 = arg0->unk4;
    spC = (u8) (temp_r1 + temp_r3);
    temp_r2 = arg0->unk8;
    subroutine_arg0.unkE = (s16) (temp_r1 + temp_r2);
    temp_r5 = temp_r3 + temp_r2;
    sp10 = temp_r5;
    sp4 = subroutine_arg0.unkE * temp_r3;
    sp8 = temp_r2 * temp_r5;
    if (spC != 0) {
        sp14 = (s32) arg0;
    } else {
        sp14 = &subroutine_arg0;
    }
    return subroutine_arg0.unkE + spC + sp10 + *sp14 + sp4;
}
