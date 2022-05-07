s32 func_00400090(s32);                             /* static */

s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 sp24;
    s32 sp20;
    s32 temp_s0_12;
    s32 temp_s0_45;
    s32 temp_t3_60;
    s32 temp_t5_82;
    s32 temp_t7_13;
    s32 temp_v0_22;
    s32 temp_v0_47;
    s32 temp_v1_42;
    s32 phi_s0_7;
    s32 phi_t0_7;
    s32 phi_v1_7;
    s32 phi_a2_7;
    s32 phi_v1_12;
    s32 phi_s0_14;
    s32 phi_t0_14;
    s32 phi_v1_14;
    s32 phi_v1_19;
    s32 phi_v1_22;
    s32 phi_v1_13;
    s32 phi_v1_20;

    temp_s0_12 = arg0 + arg1;
    temp_t7_13 = arg1 + arg2;
    sp20 = temp_t7_13;
    phi_a2_7 = arg2;
    phi_s0_7 = temp_s0_12;
    phi_t0_7 = temp_t7_13;
    if ((temp_s0_12 != 0) || (temp_t7_13 != 0) || (temp_v0_22 = func_00400090(temp_t7_13), phi_t0_7 = temp_v0_22, phi_t0_7 = temp_v0_22, (temp_v0_22 != 0)) || (phi_s0_7 = 2, phi_s0_7 = 2, (arg3 != 0))) {
        phi_v1_7 = 1;
        phi_a2_7 = arg2;
    } else {
        phi_v1_7 = -2;
        if (arg0 != 0) {
            phi_v1_7 = -1;
        }
    }
    temp_v1_42 = phi_v1_7 + phi_a2_7;
    phi_v1_12 = temp_v1_42;
    phi_s0_14 = phi_s0_7;
    phi_t0_14 = phi_t0_7;
    phi_v1_14 = temp_v1_42;
    phi_v1_13 = temp_v1_42;
    if ((phi_s0_7 != 0) && (phi_t0_7 != 0)) {
        temp_s0_45 = phi_s0_7 + phi_t0_7;
        sp24 = temp_v1_42;
        temp_v0_47 = func_00400090(temp_s0_45);
        phi_s0_14 = temp_s0_45;
        phi_t0_14 = temp_v0_47;
        if ((temp_v0_47 != 0) && (arg3 != 0)) {
            if (temp_v1_42 < 5) {
                do {
                    temp_t3_60 = (phi_v1_12 + 1) * 2;
                    phi_v1_12 = temp_t3_60;
                    phi_v1_13 = temp_t3_60;
                } while (temp_t3_60 < 5);
            }
            phi_v1_14 = phi_v1_13 + 5;
        }
    }
    phi_v1_19 = phi_v1_14;
    phi_v1_20 = phi_v1_14;
    if ((phi_s0_14 != 0) && (phi_t0_14 != 0) && (sp24 = phi_v1_14, (func_00400090(phi_s0_14 + phi_t0_14) != 0)) && (arg3 != 0)) {
        if (phi_v1_14 < 5) {
            do {
                temp_t5_82 = (phi_v1_19 + 1) * 2;
                phi_v1_19 = temp_t5_82;
                phi_v1_20 = temp_t5_82;
            } while (temp_t5_82 < 5);
        }
        phi_v1_22 = phi_v1_20 + 5;
    } else {
        phi_v1_22 = phi_v1_14 + 6;
    }
    return phi_v1_22;
}
