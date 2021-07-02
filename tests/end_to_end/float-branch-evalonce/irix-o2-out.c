f32 test(); // static
extern f32 D_410120;

f32 test(void) {
    f32 temp_f0;
    s32 temp_cond;
    s32 temp_cond_2;

    temp_cond = D_410120 < 0.0f;
    D_410120 = 5.0f;
    if (temp_cond) {
        D_410120 = 6.0f;
    }
    temp_f0 = D_410120;
    temp_cond_2 = temp_f0 < 0.0f;
    D_410120 = 3.0f;
    if (!temp_cond_2) {
        D_410120 = 7.0f;
    }
    return temp_f0;
}
