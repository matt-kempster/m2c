s32 test(s32 arg0) {
    s32 temp_a1_58;
    s32 temp_t1_12;
    s32 temp_v0_65;
    s32 temp_v1_23;
    s32 temp_v1_57;
    s32 phi_a3_4;
    s32 phi_a1_7;
    s32 phi_v0_1;
    s32 phi_v1_9;
    s32 phi_a1_6;
    s32 phi_v1_8;
    s32 phi_v1_1;
    s32 phi_v1_4;
    s32 phi_a2_4;
    s32 phi_v1_7;
    s32 phi_a2_7;
    s32 phi_a3_7;
    s32 phi_t0_7;
    s32 phi_t1_7;

    phi_v0_1 = 0;
    phi_v1_9 = 0;
    phi_v1_1 = 0;
    if (arg0 > 0) {
        do {
            phi_a1_6 = 0;
            phi_v1_8 = phi_v1_1;
            phi_v1_4 = phi_v1_1;
            phi_v1_7 = phi_v1_1;
            if (arg0 > 0) {
                temp_t1_12 = arg0 & 3;
                if (temp_t1_12 != 0) {
                    phi_a3_4 = 1;
                    phi_a2_4 = phi_v0_1 * 0;
                    do {
                        temp_v1_23 = phi_v1_4 + phi_a2_4;
                        phi_a3_4 += 1;
                        phi_a1_6 = phi_a3_4;
                        phi_v1_8 = temp_v1_23;
                        phi_v1_4 = temp_v1_23;
                        phi_a2_4 += phi_v0_1;
                        phi_v1_7 = temp_v1_23;
                    } while (temp_t1_12 != phi_a3_4);
                    if (phi_a3_4 != arg0) {
                        goto block_6;
                    }
                } else {
block_6:
                    phi_a1_7 = phi_a1_6;
                    phi_a2_7 = phi_v0_1 * phi_a1_6;
                    phi_a3_7 = phi_v0_1 * (phi_a1_6 + 1);
                    phi_t0_7 = phi_v0_1 * (phi_a1_6 + 2);
                    phi_t1_7 = phi_v0_1 * (phi_a1_6 + 3);
                    do {
                        temp_v1_57 = phi_v1_7 + phi_a2_7 + phi_a3_7 + phi_t0_7 + phi_t1_7;
                        temp_a1_58 = phi_a1_7 + 4;
                        phi_a1_7 = temp_a1_58;
                        phi_v1_8 = temp_v1_57;
                        phi_v1_7 = temp_v1_57;
                        phi_a2_7 += phi_v0_1 * 4;
                        phi_a3_7 += phi_v0_1 * 4;
                        phi_t0_7 += phi_v0_1 * 4;
                        phi_t1_7 += phi_v0_1 * 4;
                    } while (temp_a1_58 != arg0);
                }
            }
            temp_v0_65 = phi_v0_1 + 1;
            phi_v0_1 = temp_v0_65;
            phi_v1_9 = phi_v1_8;
            phi_v1_1 = phi_v1_8;
        } while (temp_v0_65 != arg0);
    }
    return phi_v1_9;
}
