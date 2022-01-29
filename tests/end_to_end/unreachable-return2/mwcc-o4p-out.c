extern s32 x;

f32 test(f32 arg8) {
loop_1:
    if ((s32) x != 2) {
        x = 1;
        goto loop_1;
    }
    return arg8;
}
