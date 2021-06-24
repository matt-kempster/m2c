s32 test(s32 arg0) {
    while (true) {
loop_3:
        func_0040010C(arg0);
        arg0 *= 2;
        if (arg0 < 4) {
            func_0040010C(arg0);
            arg0 += 1;
            goto loop_3;
        }
        break;
    }
    return arg0;
}
