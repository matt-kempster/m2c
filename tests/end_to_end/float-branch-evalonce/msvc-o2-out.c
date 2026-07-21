void test(void) {
    f32 sp0;
    f32 var_f0;
    u16 temp_fsw;

    var_f0 = 5.0f;
    sp0 = x;
    if (!(sp0 >= 0.0f)) {
        var_f0 = 6.0f;
    }
    temp_fsw = M2C_FCMP(var_f0, 0.0f);
    x = 3.0f;
    if (var_f0 >= 0.0f) {
        x = 7.0f;
    }
}
