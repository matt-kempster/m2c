extern s32 D_4100E0;

s32 test(void) {
loop_1:
    if (D_4100E0 != 2) {
        D_4100E0 = 1;
        goto loop_1;
    }
    return 2;
}
