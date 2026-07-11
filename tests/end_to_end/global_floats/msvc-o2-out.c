f32 D_410170[3];

f32 test(s32 i) {
    f32 temp_f0;
    f32 temp_f1;

    temp_f0 = D_400120[i] + D_410160[i];
    D_410170[i] = temp_f0;
    temp_f1 = ((temp_f0 * 5.67f) + D_40012C[i]) * D_410150;
    D_410150 = temp_f1;
    return temp_f1 / temp_f0;
}
