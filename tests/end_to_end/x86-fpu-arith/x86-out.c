extern f32 _g_result;

void test(f32 arg0, f32 arg1, f32 arg2) {
    _g_result = ((arg0 + arg1) * arg2) - (arg0 / arg1);
}
