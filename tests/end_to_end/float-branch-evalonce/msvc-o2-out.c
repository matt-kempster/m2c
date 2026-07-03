extern f32 _x;
static f32 _real_40c00000 = 6.0f;                   /* const */
static f32 _real_00000000 = 0.0f;                   /* const */
static f32 _real_40a00000 = 5.0f;                   /* const */

void test(void) {
    f32 sp0;
    f32 var_f0;

    var_f0 = _real_40a00000;
    sp0 = _x;
    if (!(sp0 >= _real_00000000)) {
        var_f0 = _real_40c00000;
    }
    _x = 3.0f;
    if (var_f0 >= _real_00000000) {
        _x = 7.0f;
    }
}
