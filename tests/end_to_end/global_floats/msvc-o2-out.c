f32 D_410170[3];
static f32 real_40b570a4 = 5.67f;                   /* const */

f32 test(s32 i) {
    f32 temp_f0;
    f32 temp_f1;

    temp_f0 = D_400120[i] + D_410160[i];
    D_410170[i] = temp_f0;
    temp_f1 = ((temp_f0 * real_40b570a4) + D_40012C[i]) * D_410150;
    D_410150 = temp_f1;
    return temp_f1 / temp_f0;
}
