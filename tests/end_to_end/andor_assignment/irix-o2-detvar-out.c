s32 func_00400090(s32);                             /* static */

s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 sp24;
    s32 sp20;
    s32 phi_a2;
    s32 phi_s0;
    s32 phi_t0;
    s32 phi_v1;
    s32 phi_v1_2;
    s32 phi_v1_3;
    s32 phi_v1_4;
    s32 temp_t3_60;
    s32 temp_t5_82;
    s32 temp_t7_13;
    s32 temp_v0_22;
    s32 temp_v0_47;

    phi_a2 = arg2;
    phi_s0 = arg0 + arg1;
    temp_t7_13 = arg1 + arg2;
    sp20 = temp_t7_13;
    phi_t0 = temp_t7_13;
    if ((phi_s0 != 0) || (temp_t7_13 != 0) || (temp_v0_22 = func_00400090(temp_t7_13), phi_a2 = arg2, phi_t0 = temp_v0_22, (temp_v0_22 != 0)) || (phi_s0 = 2, (arg3 != 0))) {
        phi_v1_2 = 1;
    } else {
        phi_v1_2 = -2;
        if (arg0 != 0) {
            phi_v1_2 = -1;
        }
    }
    phi_v1_3 = phi_v1_2 + phi_a2;
    if ((phi_s0 != 0) && (phi_t0 != 0)) {
        phi_s0 += phi_t0;
        sp24 = phi_v1_3;
        temp_v0_47 = func_00400090(phi_s0);
        phi_v1_3 = phi_v1_3;
        phi_t0 = temp_v0_47;
        if ((temp_v0_47 != 0) && (arg3 != 0)) {
            if (phi_v1_3 < 5) {
                do {
                    temp_t3_60 = (phi_v1_3 + 1) * 2;
                    phi_v1_3 = temp_t3_60;
                } while (temp_t3_60 < 5);
            }
            phi_v1_3 += 5;
        }
    }
    if ((phi_s0 != 0) && (phi_t0 != 0) && (sp24 = phi_v1_3, phi_v1_4 = phi_v1_3, (func_00400090(phi_s0 + phi_t0) != 0)) && (arg3 != 0)) {
        if (phi_v1_3 < 5) {
            do {
                temp_t5_82 = (phi_v1_4 + 1) * 2;
                phi_v1_4 = temp_t5_82;
            } while (temp_t5_82 < 5);
        }
        phi_v1 = phi_v1_4 + 5;
    } else {
        phi_v1 = phi_v1_3 + 6;
    }
    return phi_v1;
}
