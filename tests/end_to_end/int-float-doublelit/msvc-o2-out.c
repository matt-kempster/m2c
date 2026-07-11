s32 _ftol(f64);                                     /* extern */
static f64 real_4015333333333333 = 5.3;             /* const */
static f64 real_4014000000000000 = 5.0;             /* const */

f32 test(f32 f1, s32 i1, f32 f2, u32 i2) {
    s32 sp0;
    s32 sp4;

    globali = _ftol((f64) f1);
    sp4 = 0;
    globalf = (f32) i1;
    sp0 = i2 + 3;
    return (f32) ((f64) f2 + real_4014000000000000 + real_4015333333333333 + (f64) sp0);
}
