s32 test(s32 arg0) {
    s32 temp_ctr_2;
    s32 temp_r10;
    s32 temp_r29;
    s32 temp_r30;
    s32 temp_r30_2;
    u32 temp_ctr;
    s32 phi_r31;
    u32 phi_ctr;
    s32 phi_r29;
    s32 phi_ctr_2;
    s32 phi_r30;
    s32 phi_r29_2;
    s32 phi_r30_2;
    s32 phi_r30_3;
    s32 phi_r4;
    s32 phi_r30_4;
    s32 phi_r11;
    s32 phi_r12;

    temp_r10 = arg0 - 8;
    phi_r31 = 0;
    phi_r30 = 0;
    phi_r12 = 0;
loop_9:
    phi_r30_2 = phi_r30;
    phi_r30_4 = phi_r30;
    if (phi_r31 < arg0) {
        phi_r29 = 0;
        phi_r29_2 = 0;
        if (arg0 > 0) {
            if (arg0 > 8) {
                phi_ctr = (u32) (temp_r10 + 7) >> 3U;
                phi_r11 = 0;
                if (temp_r10 > 0) {
                    do {
                        temp_r30 = phi_r30_4 + phi_r11 + (phi_r31 * (phi_r29_2 + 1)) + (phi_r31 * (phi_r29_2 + 2)) + (phi_r31 * (phi_r29_2 + 3)) + (phi_r31 * (phi_r29_2 + 4)) + (phi_r31 * (phi_r29_2 + 5)) + (phi_r31 * (phi_r29_2 + 6)) + (phi_r31 * (phi_r29_2 + 7));
                        temp_r29 = phi_r29_2 + 8;
                        temp_ctr = phi_ctr - 1;
                        phi_ctr = temp_ctr;
                        phi_r29 = temp_r29;
                        phi_r29_2 = temp_r29;
                        phi_r30_2 = temp_r30;
                        phi_r30_4 = temp_r30;
                        phi_r11 += phi_r12;
                    } while (temp_ctr != 0);
                }
            }
            phi_r30 = phi_r30_2;
            phi_r30_3 = phi_r30_2;
            phi_r4 = phi_r29 * phi_r31;
            phi_ctr_2 = arg0 - phi_r29;
            if (phi_r29 < arg0) {
                do {
                    temp_r30_2 = phi_r30_3 + phi_r4;
                    temp_ctr_2 = phi_ctr_2 - 1;
                    phi_ctr_2 = temp_ctr_2;
                    phi_r30 = temp_r30_2;
                    phi_r30_3 = temp_r30_2;
                    phi_r4 += phi_r31;
                } while (temp_ctr_2 != 0);
            }
        }
        phi_r31 += 1;
        phi_r12 += 8;
        goto loop_9;
    }
    return phi_r30;
}
