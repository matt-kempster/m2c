extern s32 D_410150;

s32 test(s32 arg0) {
    s32 sp4;
    s32 phi_a0;
    s32 temp_a0;
    s32 temp_a0_2;
    s32 temp_a0_3;
    s32 temp_a0_4;
    s32 temp_a0_5;

    phi_a0 = arg0;
    if (arg0 == 5) {
        do {
            D_410150 = phi_a0;
            temp_a0 = phi_a0 + 1;
            D_410150 = temp_a0;
            temp_a0_2 = temp_a0 + 1;
            D_410150 = temp_a0_2;
            temp_a0_3 = temp_a0_2 + 1;
            D_410150 = temp_a0_3;
            sp4 = temp_a0_3;
            temp_a0_4 = temp_a0_3 + 1;
            D_410150 = temp_a0_4;
            D_410150 = temp_a0_4;
            temp_a0_5 = temp_a0_4 + 1;
            D_410150 = temp_a0_5;
            phi_a0 = temp_a0_5 + 1;
            D_410150 = sp4;
        } while (phi_a0 == 5);
    }
    return sp4;
}
