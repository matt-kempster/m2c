s32 test(s32 arg0) {
    s32 phi_v1;
    s32 phi_v1_2;
    s32 phi_v1_3;

    phi_v1_3 = 0;
    if (arg0 > 0) {
        phi_v1 = 0;
        do {
            goto loop_2;
            phi_v1 = phi_v1_2;
            phi_v1_3 = phi_v1_2;
        } while ((phi_v1_2 < arg0) != 0);
    }
    return phi_v1_3;
    // bug: did not emit code for node #4; contents below:
    phi_v1_2 = phi_v1 + 4;
    // bug: did not emit code for node #3; contents below:
    phi_v1_2 = phi_v1 * 2;
}
