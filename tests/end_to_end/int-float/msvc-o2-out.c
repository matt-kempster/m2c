s32 __ftol(f64);                                    /* extern */
static f32 _real_40a00000 = 5.0f;                   /* const */

f32 test(f32 f1, s32 i1, f32 f2, u32 i2) {
    s32 sp0;
    s32 sp4;

    _globali = __ftol((f64) f1);
    sp4 = 0;
    _globalf = (f32) i1;
    sp0 = i2 + 3;
    return f2 + _real_40a00000 + (f32) sp0;
}
