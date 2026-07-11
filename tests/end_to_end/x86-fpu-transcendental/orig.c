#include <math.h>

extern float g_wave;
extern float g_l2e, g_l2t, g_lg2, g_ln2;
extern float g_atan, g_logmul, g_log1pmul;
extern float g_rem, g_rem1, g_scale, g_exp2m1, g_round;

void test(float t) {
    g_wave = fabsf(sinf(t));
    g_l2e = 1.442695f;
    g_l2t = 3.321928f;
    g_lg2 = 0.30103f;
    g_ln2 = 0.693147f;
    g_atan = atan2f(t, 1.0f);
    g_logmul = t * log2f(t);
    g_log1pmul = t * log2f(t + 1.0f);
    g_rem = fmodf(t, 1.0f);
    g_rem1 = fmodf(t, 1.0f);
    g_scale = ldexpf(t, (int) t);
    g_exp2m1 = exp2f(t) - 1.0f;
    g_round = nearbyintf(t);
}
