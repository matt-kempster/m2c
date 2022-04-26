s32 test(s32 arg0) {
    s32 phi_r4;

    phi_r4 = 0;
loop_4:
    if (phi_r4 < arg0) {
        if (phi_r4 == 5) {
            phi_r4 *= 2;
        } else {
            phi_r4 += 4;
        }
        goto loop_4;
    }
    return phi_r4;
}
