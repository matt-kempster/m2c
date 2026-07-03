extern f32 _g_wave;

void test(f32 arg0) {
    _g_wave = fabsf(sinf(arg0));
}
