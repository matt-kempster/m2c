s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 temp_v0;
    s32 temp_v1;
    s32 temp_v1_2;
    s32 temp_v1_3;
    s32 temp_v1_4;
    s32 temp_v1_5;
    s32 temp_v1_6;
    s32 temp_v1_7;
    s32 phi_v0;
    s32 phi_v1;
    s32 phi_v1_2;
    s32 phi_v1_3;
    s32 phi_v1_4;
    s32 phi_v1_5;
    s32 phi_v1_6;
    s32 phi_v1_7;
    s32 phi_v1_8;
    s32 phi_v1_9;

    phi_v1_7 = 0;
    if ((arg0 != 0) && ((phi_v1_9 = 0, (arg1 != 0)) || (phi_v1_7 = 0, phi_v1_9 = 0, (arg2 != 0)))) {
loop_3:
        temp_v1 = phi_v1_9 + 1;
        phi_v1_7 = temp_v1;
        if (arg0 != 0) {
            phi_v1_9 = temp_v1;
            if ((arg1 != 0) || (phi_v1_7 = temp_v1, phi_v1_9 = temp_v1, (arg2 != 0))) {
                goto loop_3;
            }
        }
    }
    phi_v1_8 = phi_v1_7;
    if ((arg0 != 0) || ((phi_v1_4 = phi_v1_7, (arg1 != 0)) && (phi_v1_4 = phi_v1_7, phi_v1_8 = phi_v1_7, (arg2 != 0)))) {
loop_9:
        temp_v1_2 = phi_v1_8 + 1;
        phi_v1_8 = temp_v1_2;
        if (arg0 != 0) {
            goto loop_9;
        }
        phi_v1_4 = temp_v1_2;
        if (arg1 != 0) {
            phi_v1_4 = temp_v1_2;
            phi_v1_8 = temp_v1_2;
            if (arg2 != 0) {
                goto loop_9;
            }
        }
    }
    phi_v1_2 = phi_v1_4;
    phi_v1_5 = phi_v1_4;
    if (arg0 != 0) {
loop_13:
        temp_v1_3 = phi_v1_5 + 1;
        if (((arg1 != 0) && ((phi_v1_6 = temp_v1_3, (arg2 != 0)) || (phi_v1_6 = temp_v1_3, (arg3 != 0)))) || ((temp_v1_4 = temp_v1_3 + 1, phi_v1_6 = temp_v1_4, (arg1 == 0)) && ((arg2 == 0) || (phi_v1_6 = temp_v1_4, (arg3 == 0))))) {
            temp_v1_5 = temp_v1_4 + 1;
            if (((arg1 != 0) && ((phi_v1_2 = temp_v1_5, (arg2 != 0)) || (phi_v1_2 = temp_v1_5, (arg3 != 0)))) || ((temp_v1_6 = temp_v1_5 + 1, phi_v1_2 = temp_v1_6, (arg1 == 0)) && ((arg2 == 0) || (phi_v1_2 = temp_v1_6, (arg3 == 0))))) {
                phi_v1_6 = temp_v1_6 + 1;
block_26:
                phi_v1_2 = phi_v1_6;
                phi_v1_5 = phi_v1_6;
                if (arg0 != 0) {
                    goto loop_13;
                }
            }
        } else {
            goto block_26;
        }
    }
    phi_v0 = 0;
    phi_v1_3 = phi_v1_2;
    if ((arg0 != 0) || (phi_v0 = 0, phi_v1 = phi_v1_2, phi_v1_3 = phi_v1_2, (arg1 != 0))) {
loop_29:
        temp_v0 = phi_v0 + arg2 + arg3;
        temp_v1_7 = phi_v1_3 + 1;
        phi_v1 = temp_v1_7;
        if (temp_v0 < 0xA) {
            phi_v0 = temp_v0;
            phi_v1_3 = temp_v1_7;
            if ((arg0 != 0) || (phi_v0 = temp_v0, phi_v1 = temp_v1_7, phi_v1_3 = temp_v1_7, (arg1 != 0))) {
                goto loop_29;
            }
        }
    }
    return phi_v1;
}
