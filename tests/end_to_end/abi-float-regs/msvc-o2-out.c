f32 test(u32 y) {
    u32 sp0;
    u32 sp4;
    f64 var_f0;
    s32 temp_c;
    u32 var_eax;

    var_eax = y;
    var_f0 = 1.0;
    temp_c = var_eax < 0U;
    if (var_eax != 0) {
        do {
            sp0 = var_eax;
            sp4 = 0;
            var_eax -= 1;
            var_f0 *= (f64) (s64) (((u64) sp4 << 0x20) | sp0);
        } while (var_eax != 0);
    }
    return (f32) var_f0;
}
