s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 spC;
    s32 sp8;
    s32 sp4;

    spC = arg0 + arg1;
    sp8 = arg1 + arg2;
    sp4 = 0;
    if (spC != 0) {
        goto block_5;
    }
    if (!((sp8 == 0) || ((arg0 * arg1) == 0))) {
        goto block_5;
    }
    if ((arg3 != 0) && (arg0 != 0)) {
block_5:
        sp4 = arg0 + arg3;
    }
    return sp4;
}
