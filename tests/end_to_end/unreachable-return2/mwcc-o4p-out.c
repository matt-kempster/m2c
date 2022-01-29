f32 test(f32 arg8) {
loop_1:
    if ((s32) *NULL != 2) {
        *NULL = 1;
        goto loop_1;
    }
    return arg8;
}
