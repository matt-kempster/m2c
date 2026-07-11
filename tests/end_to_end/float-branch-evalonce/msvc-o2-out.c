void test(void) {
    f32 sp0;
    f32 var_f0;

    var_f0 = 5.0f;
    sp0 = x;
    if (!(sp0 >= 0.0f)) {
        var_f0 = 6.0f;
    }
    x = 3.0f;
    if (var_f0 >= 0.0f) {
        x = 7.0f;
    }
}
