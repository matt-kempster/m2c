s32 foo(s32);                                       /* static */

s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 sp14;
    s32 phi_r30;
    s32 phi_r30_2;
    s32 phi_r31;
    s32 phi_r3;
    s32 temp_r0;

    temp_r0 = arg0 + arg1;
    phi_r3 = arg1 + arg2;
    sp14 = arg3;
    phi_r31 = temp_r0;
    if ((temp_r0 != 0) || (phi_r3 != 0) || (phi_r3 = foo(), ((phi_r3 == 0) == 0)) || (phi_r31 = 2, ((sp14 == 0) == 0))) {
        phi_r30_2 = 1;
    } else if (arg0 != 0) {
        phi_r30_2 = -1;
    } else {
        phi_r30_2 = -2;
    }
    phi_r30 = phi_r30_2 + arg2;
    if ((phi_r31 != 0) && (phi_r3 != 0)) {
        phi_r31 += phi_r3;
        phi_r3 = foo(phi_r31);
        if ((phi_r3 != 0) && (sp14 != 0)) {
loop_14:
            if (phi_r30 < 5) {
                phi_r30 = (phi_r30 + 1) * 2;
                goto loop_14;
            }
            phi_r30 += 5;
        }
    }
    if ((phi_r31 != 0) && (phi_r3 != 0) && (foo(phi_r31 + phi_r3) != 0) && (sp14 != 0)) {
loop_22:
        if (phi_r30 < 5) {
            phi_r30 = (phi_r30 + 1) * 2;
            goto loop_22;
        }
        return phi_r30 + 5;
    }
    return phi_r30 + 6;
}
