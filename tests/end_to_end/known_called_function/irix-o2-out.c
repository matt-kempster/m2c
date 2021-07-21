void test(s32 x, s16 *y, s32 z, s8 *r, s16 *s, s32 *t, s32 *u) {
    s32 *phi_s0;

    phi_s0 = NULL;
loop_1:
    phi_s0 = foo(phi_s0, y, t);
    goto loop_1;
}
