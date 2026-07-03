static f32 _D_410150 = 1.23f;
static f32 _D_410160[3] = { 6.0f, 7.0f, 8.0f };
static f32 _D_410170[3];
static f32 _D_400120[3] = { 10.0f, 11.0f, 12.0f };  /* const */
static f32 _D_40012C[5] = { 14.0f, 15.0f, 16.0f, 17.0f, 18.0f }; /* const */
static f32 _real_40b570a4 = 5.67f;                  /* const */

f32 test(s32 arg0) {
    f32 temp_f0;
    f32 temp_f1;

    temp_f0 = _D_400120[arg0] + _D_410160[arg0];
    _D_410170[arg0] = temp_f0;
    temp_f1 = ((temp_f0 * _real_40b570a4) + _D_40012C[arg0]) * _D_410150;
    _D_410150 = temp_f1;
    return temp_f1 / temp_f0;
}
