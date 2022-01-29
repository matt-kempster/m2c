static ? $$25;                                      /* unable to generate initializer */

void test(void) {
    f64 temp_f0;

    temp_f0 = (f64) *NULL;
    *NULL = (f32) *NULL;
    *NULL = (f64) *NULL;
    *NULL = temp_f0;
    *NULL = (bitwise f32) &$$25;
}
