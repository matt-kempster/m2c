void test(s32 arg0, f64 arg8, f64 arg9) {
    s32 sp20;
    s32 sp24;
    f64 temp_f1;

    sp24 = arg0 ^ 0x80000000;
    sp20 = 0x43300000;
    temp_f1 = ((arg8 * ((bitwise f64) sp20 - *NULL)) + (arg8 / arg9)) - *NULL;
    if (((f32) temp_f1 < (f32) arg9) || ((f32) temp_f1 == (f32) arg9) || ((f32) temp_f1 > (f32) *NULL)) {
        *NULL = (f64) *NULL;
        return;
    }
    *NULL = (f64) *NULL;
}
