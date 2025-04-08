extern f64 global;

void test(s32 arg0, f64 farg0, f64 farg1) {
    f64 temp_f1;
    f64 var_f1;

    temp_f1 = ((farg0 * (f64) arg0) + (farg0 / farg1)) - 7.0;
    if (((f32) temp_f1 < (f32) farg1) || ((f32) temp_f1 == (f32) farg1) || ((f32) temp_f1 > (bitwise f32) 9.0)) {
        var_f1 = 5.0;
    } else {
        var_f1 = 6.0;
    }
    global = var_f1;
}
