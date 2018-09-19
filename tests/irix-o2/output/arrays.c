s32 test(s32 arg0, s32 arg1, s32 arg2) {
    s32 sp0;

    sp0 = (?32) D_400120;
    sp0.unk4 = (s16) D_400120.unk4;
    // (possible return value: (((arg2 + (arg0 * 4))->unk4 + (*(&sp0 + arg0) * *(arg1 + (arg0 * 4)))) + D_410130))
}
