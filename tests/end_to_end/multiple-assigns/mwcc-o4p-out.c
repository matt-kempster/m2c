s32 test(s32 arg0, s32 arg1) {
    s32 temp_r3;
    s32 temp_r3_2;
    s32 temp_r4;
    s32 phi_r3;
    s32 phi_r4;

    phi_r3 = arg0;
    phi_r4 = arg1;
loop_2:
    if (phi_r3 == 5) {
        *NULL = phi_r3;
        temp_r4 = phi_r3 + 3;
        *NULL = (s32) (phi_r3 + 1);
        temp_r3 = phi_r3 + 4;
        *NULL = (s32) (phi_r3 + 2);
        *NULL = temp_r4;
        *NULL = temp_r3;
        temp_r3_2 = temp_r3 + 1;
        *NULL = temp_r3;
        *NULL = temp_r3_2;
        *NULL = temp_r4;
        phi_r3 = temp_r3_2 + 1;
        phi_r4 = temp_r4;
        goto loop_2;
    }
    return phi_r4;
}
