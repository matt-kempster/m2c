s32 test(s32 arg0) {
    s32 temp_v0;
    s32 temp_v1;
    s32 phi_a3;
    s32 phi_v0;
    s32 phi_v1;
    s32 phi_v1_2;
    s32 phi_v1_3;
    s32 phi_v1_4;
    s32 phi_a2;

    phi_v0 = 0;
    phi_v1 = 0;
    phi_v1_3 = 0;
    if (arg0 > 0) {
        do {
            goto loop_1;
            temp_v0 = phi_v0 + 1;
            phi_v0 = temp_v0;
            phi_v1 = phi_v1_2;
            phi_v1_3 = phi_v1_2;
        } while (temp_v0 != arg0);
    }
    return phi_v1;
    // bug: did not emit code for node #4; contents below:
    temp_v1 = phi_v1_4 + phi_a2;
    phi_a3 += 1;
    phi_v1_2 = temp_v1;
    phi_v1_4 = temp_v1;
    phi_a2 += phi_v0;
    // bug: did not emit code for node #2; contents below:
    // bug: did not emit code for node #3; contents below:
    phi_a3 = 1;
    phi_v1_4 = phi_v1_3;
    phi_a2 = phi_v0 * 0;
}
