f64 test(f32 arg0, s32 arg2, f64 arg4) {
    f64 sp8;
    f64 sp0;

    sp8 = ((f64) arg2 * (f64) arg0) + ((f64) arg0 / arg4);
    sp8 = sp8 - 7.0;
    if ((sp8 < arg4) || (sp8 == arg4) || (9.0 < sp8)) {
        sp0 = 0.0;
    } else {
        sp0 = 0.0;
    }
    D_410180 = sp0;
    return sp0;
}
