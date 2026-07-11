s32 _ftol(f64);                                     /* extern */

f32 test(f32 f1, s32 i1, f32 f2, u32 i2) {
    s32 sp0;
    s32 sp4;

    globali = _ftol((f64) f1);
    sp4 = 0;
    globalf = (f32) i1;
    sp0 = i2 + 3;
    return (f32) ((f64) f2 + 5.0 + 5.3 + (f64) sp0);
}
