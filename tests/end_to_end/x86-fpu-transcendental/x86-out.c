extern f32 g_wave;

void test(f32 arg0) {
    g_wave = fabsf(sinf(arg0));
}
