s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 sp24;
    s32 sp20;
    s32 sp1C;
    s32 temp_ret;
    s32 temp_t0;
    s32 temp_t1;
    s32 temp_t2;
    s32 temp_t6;
    s32 temp_v1;
    s32 phi_t1;
    s32 phi_t0;
    s32 phi_v1;
    s32 phi_v1_2;
    s32 phi_v1_3;
    s32 phi_v1_4;

    temp_t1 = arg0 + arg1;
    temp_t6 = arg1 + arg2;
    sp1C = temp_t6;
    phi_t0 = temp_t6;
    if ((((temp_t1 != 0) || (phi_t0 = temp_t6, (temp_t6 != 0))) || (sp24 = 0, sp20 = temp_t1, temp_ret = func_00400090(temp_t6), temp_t0 = temp_ret, phi_t0 = temp_t0, (temp_ret != 0))) || (phi_t1 = temp_t1, phi_t0 = temp_t0, phi_v1 = sp24, phi_t0 = temp_t0, (arg3 != 0))) {
        phi_v1 = 1;
    } else {

    }
    phi_v1_3 = phi_v1;
    if (phi_t1 != 0) {
        phi_v1_3 = phi_v1;
        if (phi_t0 != 0) {
            sp24 = phi_v1;
            temp_v1 = phi_v1;
            phi_v1_3 = temp_v1;
            if (func_00400090(phi_t1 + phi_t0) != 0) {
                phi_v1_3 = temp_v1;
                if (arg3 != 0) {
                    phi_v1_2 = temp_v1;
                    phi_v1_4 = temp_v1;
                    if (temp_v1 < 5) {
loop_10:
                        temp_t2 = (phi_v1_2 + 1) * 2;
                        phi_v1_2 = temp_t2;
                        phi_v1_4 = temp_t2;
                        if (temp_t2 < 5) {
                            goto loop_10;
                        }
                    }
                    phi_v1_3 = phi_v1_4 + 5;
                }
            }
        }
    }
    return phi_v1_3;
}
