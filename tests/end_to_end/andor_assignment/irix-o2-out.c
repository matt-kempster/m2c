s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 sp28;
    s32 sp24;
    s32 sp20;
    s32 sp1C;
    s32 temp_ret;
    s32 temp_t0;
    s32 temp_t6;
    s32 temp_v1;
    s32 phi_v1;
    s32 phi_t0;
    s32 phi_t0_2;

    temp_t6 = arg1 + arg2;
    temp_v1 = arg0 + arg1;
    sp1C = temp_t6;
    sp28 = temp_t6;
    if ((((temp_v1 != 0) || (temp_t6 != 0)) || (sp20 = temp_v1, sp24 = 0, temp_ret = func_00400090(temp_t6), sp28 = temp_ret, (temp_ret != 0))) || (phi_v1 = temp_v1, phi_t0 = sp24, (arg3 != 0))) {
        phi_t0 = 1;
    } else {

    }
    phi_t0_2 = phi_t0;
    if (phi_v1 != 0) {
        phi_t0_2 = phi_t0;
        if (sp28 != 0) {
            sp24 = phi_t0;
            temp_t0 = phi_t0;
            phi_t0_2 = temp_t0;
            if (func_00400090(sp28) != 0) {
                phi_t0_2 = temp_t0;
                if (arg3 != 0) {
                    phi_t0_2 = 2;
                }
            }
        }
    }
    return phi_t0_2;
}
