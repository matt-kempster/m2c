f32 test(s32 arg0) {
    f64 temp_ft1;
    f64 temp_ft1_2;
    f64 temp_ft2;
    f64 temp_ft4;
    f64 temp_ft5;
    f64 temp_fv1;
    f64 temp_fv1_2;
    s32 temp_a1;
    s32 temp_t6;
    s32 temp_t7;
    s32 temp_t8;
    s32 temp_v0;
    s32 temp_v0_2;
    s32 phi_v0;
    s32 phi_v0_2;
    f64 phi_fv1;
    f64 phi_fv1_2;
    f64 phi_ft1;
    f64 phi_fv1_3;
    f64 phi_ft4;
    f64 phi_ft2;
    f64 phi_ft5;
    f64 phi_ft1_2;

    phi_v0 = arg0;
    phi_v0_2 = arg0;
    phi_fv1 = 0.0;
    phi_fv1_2 = 0.0;
    phi_fv1_3 = 0.0;
    if (arg0 != 0) {
        temp_a1 = -(arg0 & 3);
        if (temp_a1 != 0) {
            do {
                temp_ft1 = (f64) phi_v0;
                phi_ft1 = temp_ft1;
                if (phi_v0 < 0) {
                    phi_ft1 = temp_ft1 + 4294967296.0;
                }
                temp_fv1 = phi_fv1_2 * phi_ft1;
                temp_v0 = phi_v0 - 1;
                phi_v0 = temp_v0;
                phi_v0_2 = temp_v0;
                phi_fv1 = temp_fv1;
                phi_fv1_2 = temp_fv1;
                phi_fv1_3 = temp_fv1;
            } while ((temp_a1 + arg0) != temp_v0);
            if (temp_v0 != 0) {
                goto loop_6;
            }
        } else {
            do {
loop_6:
                temp_t6 = phi_v0_2 - 1;
                temp_ft4 = (f64) phi_v0_2;
                phi_ft4 = temp_ft4;
                if (phi_v0_2 < 0) {
                    phi_ft4 = temp_ft4 + 4294967296.0;
                }
                temp_t7 = phi_v0_2 - 2;
                temp_t8 = phi_v0_2 - 3;
                temp_ft2 = (f64) temp_t6;
                phi_ft2 = temp_ft2;
                if (temp_t6 < 0) {
                    phi_ft2 = temp_ft2 + 4294967296.0;
                }
                temp_v0_2 = phi_v0_2 - 4;
                temp_ft5 = (f64) temp_t7;
                phi_v0_2 = temp_v0_2;
                phi_ft5 = temp_ft5;
                if (temp_t7 < 0) {
                    phi_ft5 = temp_ft5 + 4294967296.0;
                }
                temp_ft1_2 = (f64) temp_t8;
                phi_ft1_2 = temp_ft1_2;
                if (temp_t8 < 0) {
                    phi_ft1_2 = temp_ft1_2 + 4294967296.0;
                }
                temp_fv1_2 = phi_fv1_3 * phi_ft4 * phi_ft2 * phi_ft5 * phi_ft1_2;
                phi_fv1 = temp_fv1_2;
                phi_fv1_3 = temp_fv1_2;
            } while (temp_v0_2 != 0);
        }
    }
    return (f32) phi_fv1;
}
