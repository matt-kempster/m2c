s32 test(s8 *arg0, s32 arg1) {
    s32 temp_a3;
    s32 temp_v0;
    s32 temp_v0_2;
    void *temp_v1;
    s8 *phi_v1;
    s32 phi_v0;
    void *phi_v1_2;
    s32 phi_v0_2;
    s32 phi_return;
    s32 phi_v0_3;

    phi_return = 0;
    if (arg1 > 0) {
        temp_a3 = arg1 & 3;
        phi_v0_3 = 0;
        if (temp_a3 == 0) {
            goto block_5;
        }
        phi_v1 = arg0;
        phi_v0 = 0;
loop_3:
        temp_v0 = phi_v0 + 1;
        *phi_v1 = (u8)0;
        phi_v1 += 1;
        phi_v0 = temp_v0;
        if (temp_a3 != temp_v0) {
            goto loop_3;
        }
        phi_return = temp_v0;
        phi_v0_3 = temp_v0;
        if (temp_v0 != arg1) {
block_5:
            phi_v1_2 = arg0 + phi_v0_3;
            phi_v0_2 = phi_v0_3;
loop_6:
            temp_v0_2 = phi_v0_2 + 4;
            phi_v1_2->unk1 = (u8)0;
            phi_v1_2->unk2 = (u8)0;
            phi_v1_2->unk3 = (u8)0;
            temp_v1 = phi_v1_2 + 4;
            temp_v1->unk-4 = (u8)0;
            phi_v1_2 = temp_v1;
            phi_v0_2 = temp_v0_2;
            phi_return = temp_v0_2;
            if (temp_v0_2 != arg1) {
                goto loop_6;
            }
        }
    }
    return phi_return;
}
