u32 _ftol(f64);                                     /* extern */

void test(void) {
    u32 sp0;
    u32 sp4;

    u = _ftol((f64) flt);
    u = _ftol(dbl);
    sp0 = u;
    sp4 = 0;
    dbl = (f64) (s64) (((u64) sp4 << 0x20) | sp0);
    sp0 = u;
    sp4 = 0;
    flt = (f32) (s64) (((u64) sp4 << 0x20) | sp0);
}
