f64 test(f64 arg0, s32 arg2, f64 arg4); // static
extern f64 D_410150;

f64 test(f64 arg0, s32 arg2, f64 arg4) {
    f64 temp_f0;

    temp_f0 = (((f64) arg2 * arg0) + (arg0 / arg4)) - 7.0;
    if (!(temp_f0 < arg4)) {
        if ((temp_f0 == arg4) || (temp_f0 > 9.0)) {
block_4:
        }
    } else {
        goto block_4;
    }
    D_410150 = 0.0;
    return 0.0;
}
