static s16 globalArray[4];

s32 test(s32 arg0, s32 arg1, s32 arg2) {
    s32 sp14;
    u16 sp18;
    s32 temp_r10;

    temp_r10 = arg0 * 4;
    sp14 = *NULL;
    sp18 = (u16) *NULL;
    return globalArray[arg0] + (((s8) *(&sp14 + arg0) * *(arg1 + temp_r10)) + (arg2 + temp_r10)->unk4);
}
