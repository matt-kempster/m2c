s32 test(s32 arg0) {
    s32 phi_a1;
    s32 phi_a2;
    s32 phi_a2_2;
    s32 phi_a3;
    s32 phi_a3_2;
    s32 phi_t0;
    s32 phi_t1;
    s32 phi_v0;
    s32 phi_v1;
    s32 temp_t1;

    phi_v0 = 0;
    phi_v1 = 0;
    if (arg0 > 0) {
        do {
            phi_a1 = 0;
            if (arg0 > 0) {
                temp_t1 = arg0 & 3;
                if (temp_t1 != 0) {
                    phi_a3 = 1;
                    phi_a2 = phi_v0 * 0;
                    do {
                        phi_a1 = phi_a3;
                        phi_v1 += phi_a2;
                        phi_a2 += phi_v0;
                        phi_a3 += 1;
                    } while (temp_t1 != phi_a3);
                    if (phi_a3 != arg0) {
                        goto block_6;
                    }
                } else {
block_6:
                    phi_a2_2 = phi_v0 * phi_a1;
                    phi_a3_2 = phi_v0 * (phi_a1 + 1);
                    phi_t0 = phi_v0 * (phi_a1 + 2);
                    phi_t1 = phi_v0 * (phi_a1 + 3);
                    do {
                        phi_v1 = phi_v1 + phi_a2_2 + phi_a3_2 + phi_t0 + phi_t1;
                        phi_a1 += 4;
                        phi_t1 += phi_v0 * 4;
                        phi_t0 += phi_v0 * 4;
                        phi_a3_2 += phi_v0 * 4;
                        phi_a2_2 += phi_v0 * 4;
                    } while (phi_a1 != arg0);
                }
            }
            phi_v0 += 1;
        } while (phi_v0 != arg0);
    }
    return phi_v1;
}
