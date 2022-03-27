f32 test(s32 arg0) {
    f64 sp8;
    s32 sp4;
    f64 temp_ft3;
    s32 temp_t9;
    f64 phi_ft3;

    sp8 = 1.0;
    sp4 = arg0;
    if (sp4 != 0) {
        do {
            temp_ft3 = (f64) sp4;
            phi_ft3 = temp_ft3;
            if (sp4 < 0) {
                phi_ft3 = temp_ft3 + 4294967296.0;
            }
            sp8 *= phi_ft3;
            temp_t9 = sp4 - 1;
            sp4 = temp_t9;
        } while (temp_t9 != 0);
    }
    return (f32) sp8;
}
