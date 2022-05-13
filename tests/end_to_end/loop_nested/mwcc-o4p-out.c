s32 test(s32 arg0) {
    s32 phi_ctr_2;
    s32 phi_r11;
    s32 phi_r12;
    s32 phi_r29;
    s32 phi_r30;
    s32 phi_r31;
    s32 phi_r4;
    s32 temp_r10;
    u32 phi_ctr;

    temp_r10 = arg0 - 8;
    phi_r12 = 0;
    phi_r31 = 0;
    phi_r30 = 0;
loop_9:
    if (phi_r31 < arg0) {
        phi_r29 = 0;
        if (arg0 > 0) {
            if (arg0 > 8) {
                phi_ctr = (u32) (temp_r10 + 7) >> 3U;
                phi_r11 = 0;
                if (temp_r10 > 0) {
                    do {
                        phi_r30 = phi_r30 + phi_r11 + (phi_r31 * (phi_r29 + 1)) + (phi_r31 * (phi_r29 + 2)) + (phi_r31 * (phi_r29 + 3)) + (phi_r31 * (phi_r29 + 4)) + (phi_r31 * (phi_r29 + 5)) + (phi_r31 * (phi_r29 + 6)) + (phi_r31 * (phi_r29 + 7));
                        phi_r11 += phi_r12;
                        phi_r29 += 8;
                        phi_ctr -= 1;
                    } while (phi_ctr != 0);
                }
            }
            phi_r4 = phi_r29 * phi_r31;
            phi_ctr_2 = arg0 - phi_r29;
            if (phi_r29 < arg0) {
                do {
                    phi_r30 += phi_r4;
                    phi_r4 += phi_r31;
                    phi_ctr_2 -= 1;
                } while (phi_ctr_2 != 0);
            }
        }
        phi_r12 += 8;
        phi_r31 += 1;
        goto loop_9;
    }
    return phi_r30;
}
