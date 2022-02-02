f32 test(s32 arg0, s32 arg1, f32 arg8) {
    s32 temp_ctr_2;
    s32 temp_r5;
    s32 temp_r6;
    u32 temp_ctr;
    void *temp_r5_2;
    s32 phi_r6;
    u32 phi_ctr;
    s32 phi_r6_2;
    s8 *phi_r3;
    s32 phi_ctr_2;

    phi_r6 = 0;
    phi_r6_2 = 0;
    if (arg1 > 0) {
        temp_r5 = arg1 - 8;
        if (arg1 > 8) {
            phi_ctr = (u32) (temp_r5 + 7) >> 3U;
            if (temp_r5 > 0) {
                do {
                    temp_r5_2 = arg0 + phi_r6;
                    temp_r5_2->unk0 = 0;
                    temp_r6 = phi_r6 + 8;
                    temp_r5_2->unk1 = 0;
                    temp_r5_2->unk2 = 0;
                    temp_r5_2->unk3 = 0;
                    temp_r5_2->unk4 = 0;
                    temp_r5_2->unk5 = 0;
                    temp_r5_2->unk6 = 0;
                    temp_r5_2->unk7 = 0;
                    temp_ctr = phi_ctr - 1;
                    phi_r6 = temp_r6;
                    phi_ctr = temp_ctr;
                    phi_r6_2 = temp_r6;
                } while (temp_ctr != 0);
            }
        }
        phi_r3 = arg0 + phi_r6_2;
        phi_ctr_2 = arg1 - phi_r6_2;
        if (phi_r6_2 < arg1) {
            do {
                *phi_r3 = 0;
                temp_ctr_2 = phi_ctr_2 - 1;
                phi_r3 += 1;
                phi_ctr_2 = temp_ctr_2;
            } while (temp_ctr_2 != 0);
            return arg8;
        }
        /* Duplicate return node #7. Try simplifying control flow for better match */
        return arg8;
    }
    return arg8;
}
