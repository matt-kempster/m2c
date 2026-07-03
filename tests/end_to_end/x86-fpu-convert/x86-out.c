extern s32 _g_i;

void test(s32 arg0, f32 arg1) {
    s64 sp0;

    sp0 = (s64) ((f32) arg0 * arg1);
    _g_i = (s32) sp0;
}
