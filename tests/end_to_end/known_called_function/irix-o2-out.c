void test(s32 x, short *y, s32 z) {
    int *phi_s0;

    phi_s0 = NULL;
loop_1:
    phi_s0 = foo(phi_s0, y);
    goto loop_1;
}
