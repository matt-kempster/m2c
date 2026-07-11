u32 __ftol(f64);                                    /* extern */

void test(void) {
    u32 sp0;
    u32 sp4;

    _u = __ftol((f64) _flt);
    _u = __ftol(_dbl);
    sp0 = _u;
    sp4 = 0;
    _dbl = (f64) (s64) (((u64) sp4 << 0x20) | sp0);
    sp0 = _u;
    sp4 = 0;
    _flt = (f32) (s64) (((u64) sp4 << 0x20) | sp0);
}
