s32 __ftol(f64);                                    /* extern */
extern f32 _globalf;
extern s32 _globali;
static f64 _real_4015333333333333 = 5.3;            /* const */
static f64 _real_4014000000000000 = 5.0;            /* const */

f64 test(f32 arg0, s32 arg1, f32 arg2, s32 arg3) {
    s32 sp0;
    s32 sp4;

    _globali = __ftol((f64) arg0);
    sp4 = 0;
    _globalf = (f32) arg1;
    sp0 = arg3 + 3;
    return (f64) arg2 + _real_4014000000000000 + _real_4015333333333333 + (f64) sp0;
}
