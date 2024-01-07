extern f64 D_410150;

f64 test(f64 arg0, s32 arg2, f64 arg4) {
    f64 temp_f0;
    u32 phi_f3;

    temp_f0 = (((f64) arg2 * arg0) + (arg0 / arg4)) - 7.0;
    if ((temp_f0 < arg4) || (temp_f0 == arg4) || (temp_f0 > 9.0)) {
        phi_f3 = 0x40140000U;
    } else {
        phi_f3 = 0x40180000U;
    }
    D_410150 = GLUE_F64(phi_f3, 0U);
    return GLUE_F64(phi_f3, 0U);
}
