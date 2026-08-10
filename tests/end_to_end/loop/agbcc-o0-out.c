void test(s32 arg0, s32 arg1) {
    s32 sp0;
    s32 sp4;
    s32 sp8;

    sp0 = arg0;
    sp4 = arg1;
    sp8 = 0;
loop_1:
    if (sp8 < sp4) {
        *(sp0 + sp8) = 0;
        sp8 += 1;
        goto loop_1;
    }
}
