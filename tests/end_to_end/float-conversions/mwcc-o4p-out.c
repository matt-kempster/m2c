u32 __cvt_fp2unsigned(f32);                         /* extern */
extern f64 dbl;
extern f32 flt;
extern u32 u;

void test(void) {
    u = __cvt_fp2unsigned(flt);
    u = __cvt_fp2unsigned((f32) dbl);
    dbl = (f64) u;
    flt = (f32) u;
}
