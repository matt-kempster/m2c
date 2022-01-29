static ? $$25;                                      /* unable to generate initializer */

void test(void) {
    f64 temp_f0;

    temp_f0 = *NULL;
    *NULL = (f32) *MIPS2C_ERROR(Read from unset register $r0);
    *NULL = (f64) *NULL;
    *NULL = temp_f0;
    *NULL = &$$25;
}
