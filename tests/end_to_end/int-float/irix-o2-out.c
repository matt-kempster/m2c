extern s32 D_410110;
extern f32 D_410114;

f32 test(f32 arg0, s32 arg1, f32 arg2, s32 arg3) {
    f32 temp_f18;
    s32 temp_a3;
    f32 phi_f18;

    temp_a3 = arg3 + 3;
    D_410110 = (s32) arg0;
    D_410114 = (f32) arg1;
    temp_f18 = (f32) temp_a3;
    phi_f18 = temp_f18;
    if (temp_a3 < 0) {
        phi_f18 = temp_f18 + 4294967296.0f;
    }
    return phi_f18 + (arg2 + 5.0f);
}
