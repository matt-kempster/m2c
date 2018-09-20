s32 test(s32 arg0, s32 arg1, s32 arg2) {
    *sp = (?32) D_400120;
    sp->unk4 = (s16) D_400120.unk4;
    // (possible return value: (((arg2 + (arg0 * 4))->unk4 + (*(sp + arg0) * *(arg1 + (arg0 * 4)))) + *(&D_410130 + (arg0 * 2))))
}
