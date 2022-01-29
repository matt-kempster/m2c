s32 __cvt_fp2unsigned(f32);                         /* extern */

void test(void) {
    s32 sp8;
    s32 spC;
    s32 sp10;
    s32 sp14;
    f64 temp_f1;

    *NULL = __cvt_fp2unsigned(*saved_reg_lr);
    *NULL = __cvt_fp2unsigned((f32) (bitwise f64) *NULL);
    temp_f1 = (bitwise f64) *NULL;
    sp14 = *NULL;
    sp10 = 0x43300000;
    *NULL = (f64) ((bitwise f64) sp10 - temp_f1);
    spC = *NULL;
    sp8 = 0x43300000;
    *NULL = (f32) ((f32) (bitwise f64) sp8 - (f32) temp_f1);
}
