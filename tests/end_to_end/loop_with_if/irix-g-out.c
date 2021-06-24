s32 test(s32 arg0) {
    s32 sp4;

    sp4 = 0;
    if (sp4 < arg0) {
        do {
        goto loop_1;
        } while ((sp4 < arg0) != 0);
    }
    return sp4;
    // bug: did not emit code for node #2; contents below:
    sp4 *= 2;
    // bug: did not emit code for node #3; contents below:
    sp4 += 4;
}
