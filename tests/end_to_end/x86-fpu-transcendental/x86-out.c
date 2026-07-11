extern f32 g_atan;
extern f32 g_exp2m1;
extern f32 g_l2e;
extern f32 g_l2t;
extern f32 g_lg2;
extern f32 g_ln2;
extern f32 g_log1pmul;
extern f32 g_logmul;
extern f32 g_rem;
extern f32 g_rem1;
extern f32 g_round;
extern f32 g_scale;
extern f32 g_wave;

void test(f32 arg0) {
    g_wave = fabsf(sinf(arg0));
    g_l2e = M2C_LOG2E();
    g_l2t = M2C_LOG2T();
    g_lg2 = M2C_LOG10_2();
    g_ln2 = M2C_LN2();
    g_atan = atan2f(arg0, 1.0f);
    g_logmul = arg0 * log2f(arg0);
    g_log1pmul = arg0 * log2f(arg0 + 1.0f);
    g_rem = fmodf(arg0, 1.0f);
    g_rem1 = fmodf(arg0, 1.0f);
    g_scale = ldexpf(arg0, (s32) arg0);
    g_exp2m1 = exp2f(arg0) - 1.0f;
    g_round = M2C_RNDINT(arg0);
}
