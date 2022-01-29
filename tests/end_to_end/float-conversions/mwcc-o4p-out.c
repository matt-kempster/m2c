s32 __cvt_fp2unsigned(f32);                         /* extern */

void test(void) {
    s32 sp8;
    f32 spC;
    s32 sp10;
    f32 sp14;
    f64 temp_f1;

    *NULL = (bitwise f32) __cvt_fp2unsigned(*NULL);
    *NULL = (bitwise f32) __cvt_fp2unsigned((f32) (f64) *NULL);
    temp_f1 = (f64) *NULL;
    sp14 = *NULL;
    sp10 = 0x43300000;
    *NULL = (f64) ((bitwise f64) sp10 - temp_f1);
    spC = *NULL;
    sp8 = 0x43300000;
    *NULL = (f32) ((f32) (bitwise f64) sp8 - (f32) temp_f1);
}
