void test(s8 *arg0, s32 arg1) {
    s32 phi_v0;
    s32 temp_a3;
    s8 *phi_v1;
    void *phi_v1_2;

    phi_v0 = 0;
    if (arg1 > 0) {
        temp_a3 = arg1 & 3;
        if (temp_a3 != 0) {
            phi_v1 = arg0;
            do {
                phi_v0 += 1;
                *phi_v1 = 0;
                phi_v1 += 1;
            } while (temp_a3 != phi_v0);
            phi_v1_2 = arg0 + phi_v0;
            if (phi_v0 != arg1) {
                goto block_5;
            }
        } else {
block_5:
            phi_v1_2 = arg0 + phi_v0;
            do {
                phi_v0 += 4;
                phi_v1_2->unk1 = 0;
                phi_v1_2->unk2 = 0;
                phi_v1_2->unk3 = 0;
                phi_v1_2 += 4;
                phi_v1_2->unk-4 = 0;
            } while (phi_v0 != arg1);
        }
    }
}
