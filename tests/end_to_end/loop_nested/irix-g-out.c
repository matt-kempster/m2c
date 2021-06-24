s32 test(s32 arg0) {
    s32 spC;
    s32 sp8;
    s32 sp4;

    spC = 0;
    sp8 = 0;
    if (spC < arg0) {
        do {
            goto loop_1;
            spC += 1;
        } while ((spC < arg0) != 0);
    }
    return sp8;
    // bug: did not emit code for node #2; contents below:
    sp8 += spC * sp4;
    sp4 += 1;
}
