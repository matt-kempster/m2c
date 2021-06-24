s32 test(s8 *arg0, s32 arg1) {
    s32 temp_v0;
    s8 *phi_v1;
    s32 phi_v0;
    s32 phi_return;

    phi_return = 0;
    if (arg1 > 0) {
        phi_v1 = arg0;
        phi_v0 = 0;
        do {
            temp_v0 = phi_v0 + 1;
            *phi_v1 = (u8)0;
            phi_v1 += 1;
            phi_v0 = temp_v0;
            phi_return = temp_v0;
        } while (arg1 != temp_v0);
    }
    return phi_return;
    // bug: did not emit code for node #3; contents below:
    temp_v0 = phi_v0 + 1;
    *phi_v1 = (u8)0;
    phi_v1 += 1;
    phi_v0 = temp_v0;
    phi_return = temp_v0;
}
