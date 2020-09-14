s32 test(s32 arg0) {
    goto block_3;
loop_2:
    func_0040010C(arg0);
    arg0 = arg0 + 1;
block_3:
    func_0040010C(arg0);
    arg0 = arg0 * 2;
    if (arg0 < 4) {
        goto loop_2;
    }
    return arg0;
}
