void test(s32 arg0, s32 arg1, s32 arg4) {
    unksp0 = arg0;
    unksp4 = arg1;
    arg4 = 0;
loop_1:
    if (arg4 < unksp4) {
        *(unksp0 + arg4) = 0;
        arg4 += 1;
        goto loop_1;
    }
}
