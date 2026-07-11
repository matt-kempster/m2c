f32 _D_410170[3];
static f32 _real_40b570a4 = 5.67f;                  /* const */

f32 test(s32 i) {
    f32 temp_f0;
    f32 temp_f1;

    temp_f0 = _D_400120[i] + _D_410160[i];
    _D_410170[i] = temp_f0;
    temp_f1 = ((temp_f0 * _real_40b570a4) + _D_40012C[i]) * _D_410150;
    _D_410150 = temp_f1;
    return temp_f1 / temp_f0;
}
