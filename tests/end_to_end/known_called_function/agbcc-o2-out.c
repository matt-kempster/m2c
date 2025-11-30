void test(s32 x, s16* y, s32 z, s8* r, s16* s, s32* t, s32* u) {
    s32* var_r0;

    var_r0 = NULL;
loop_1:
    var_r0 = foo(var_r0, y, t);
    goto loop_1;
}
