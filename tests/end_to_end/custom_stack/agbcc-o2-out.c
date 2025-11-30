struct _m2c_stack_test {
    /* 0x00 */ char pad0[0x14];
};                                                  /* size = 0x14 */

? frob(s32 **);                                     /* static */

s32 test(s32 *arg0, s32 *arg4) {
    s32 temp_r1;
    s32 temp_r2;
    s32 temp_r3;
    s32 temp_r5;

    frob(&unkspC);
    frob(&unksp0 + 0xE);
    frob(&unksp10);
    frob(&arg4);
    frob(&unksp0);
    temp_r1 = arg0->unk0;
    temp_r3 = arg0->unk4;
    unkspC = (u8) (temp_r1 + temp_r3);
    temp_r2 = arg0->unk8;
    unksp0.unkE = (s16) (temp_r1 + temp_r2);
    temp_r5 = temp_r3 + temp_r2;
    unksp10 = temp_r5;
    unksp0 = unkspC * temp_r1;
    unksp4 = unksp0.unkE * temp_r3;
    unksp8 = temp_r2 * temp_r5;
    if (unkspC != 0) {
        arg4 = arg0;
    } else {
        arg4 = &unksp0;
    }
    return unksp0.unkE + unkspC + unksp10 + *arg4 + unksp4;
}
