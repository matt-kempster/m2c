void test(s32 x, s16 *y, s32 z, s8 *r, s16 *s, s32 *t, s32 *u) {
    s32 *var_eax;

    var_eax = NULL;
loop_1:
    var_eax = foo(var_eax, y, t);
    goto loop_1;
}
