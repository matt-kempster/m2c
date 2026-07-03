s32 __ftol(f64);                                    /* extern */
extern f64 _dbl;
extern f32 _flt;
extern s32 _u;

void test(void) {
    s32 sp0;
    s32 sp4;

    _u = __ftol((f64) _flt);
    _u = __ftol(_dbl);
    sp0 = _u;
    sp4 = 0;
    _dbl = (f64) (s64) sp0;
    sp0 = _u;
    sp4 = 0;
    _flt = (f32) (s64) sp0;
}
