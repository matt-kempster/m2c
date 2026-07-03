s32 __ftol(f64);                                    /* extern */
extern f32 _globalf;
extern s32 _globali;
static f32 _real_40a00000 = 5.0f;                   /* const */

f32 test(f32 arg0, s32 arg1, f32 arg2, s32 arg3) {
    s32 sp0;
    s32 sp4;

    _globali = __ftol((f64) arg0);
    sp4 = 0;
    _globalf = (f32) arg1;
    sp0 = arg3 + 3;
    return arg2 + _real_40a00000 + (f32) sp0;
}
