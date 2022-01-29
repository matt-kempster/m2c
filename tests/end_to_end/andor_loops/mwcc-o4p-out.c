s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 temp_r7;
    s32 temp_r7_2;
    s32 temp_r7_3;
    s32 temp_r7_4;
    s32 phi_r5;
    s32 phi_r7;
    s32 phi_r7_2;
    s32 phi_r7_3;
    s32 phi_r7_4;

    phi_r7_4 = 0;
loop_2:
    phi_r7_3 = phi_r7_4;
    if (arg0 != 0) {
        if ((arg1 == 0) && (arg2 == 0)) {

        } else {
            phi_r7_4 += 1;
            goto loop_2;
        }
    }
loop_7:
    phi_r7_2 = phi_r7_3;
    if (arg0 != 0) {
block_6:
        phi_r7_3 += 1;
        goto loop_7;
    }
    if (arg1 != 0) {
        if (arg2 == 0) {

        } else {
            goto block_6;
        }
    }
loop_24:
    phi_r7 = phi_r7_2;
    if (arg0 != 0) {
        temp_r7 = phi_r7_2 + 1;
        phi_r7_2 = temp_r7;
        if (((arg1 == 0) || ((arg2 == 0) && (arg3 == 0))) && (temp_r7_2 = temp_r7 + 1, phi_r7_2 = temp_r7_2, ((arg1 == 0) != 0)) && ((arg2 == 0) || (arg3 == 0))) {
            temp_r7_3 = temp_r7_2 + 1;
            phi_r7 = temp_r7_3;
            if ((arg1 == 0) || ((arg2 == 0) && (arg3 == 0))) {
                temp_r7_4 = temp_r7_3 + 1;
                phi_r7 = temp_r7_4;
                if ((arg1 == 0) && ((arg2 == 0) || (arg3 == 0))) {
                    phi_r7_2 = temp_r7_4 + 1;
                    goto loop_24;
                }
            }
        } else {
            goto loop_24;
        }
    }
    phi_r5 = 0;
loop_27:
    if ((phi_r5 < 0xA) && ((arg0 != 0) || (arg1 != 0))) {
        phi_r5 += arg2 + arg3;
        phi_r7 += 1;
        goto loop_27;
    }
    return phi_r7;
}
