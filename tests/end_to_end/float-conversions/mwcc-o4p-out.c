s32 __cvt_fp2unsigned(f32);                         /* extern */
extern f64 dbl;
extern f32 flt;
extern s32 u;

void test(void) {
    s32 sp8;
    s32 spC;
    s32 sp10;
    s32 sp14;

    u = __cvt_fp2unsigned(flt);
    u = __cvt_fp2unsigned((f32) dbl);
    sp14 = u;
    sp10 = 0x43300000;
    dbl = (bitwise f64) sp10 - 4503599627370496.0;
    spC = u;
    sp8 = 0x43300000;
    flt = (f32) (bitwise f64) sp8 - (f32) 4503599627370496.0;
}
