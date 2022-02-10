? test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 phi_r7;
    s32 phi_r5;
    ? phi_r8;

    phi_r5 = arg2;
    phi_r8 = 0;
    if (((arg0 + arg1) != 0) || (((s32) (arg1 + arg2) != 0) && ((arg0 * arg1) != 0)) || ((arg3 != 0) && (arg0 != 0))) {
        phi_r8 = 1;
    }
    if ((arg0 != 0) && ((arg1 != 0) || (arg2 != 0)) && ((arg3 != 0) || ((arg0 + 1) != 0))) {
        phi_r8 = 2;
    }
    if (((arg0 != 0) && (arg3 != 0)) || (((arg1 != 0) || (arg2 != 0)) && ((arg0 + 1) != 0))) {
        phi_r8 = 3;
    }
    if ((arg0 != 0) && (arg1 != 0) && ((arg2 != 0) || (arg3 != 0)) && (((arg0 + 1) != 0) || ((arg1 + 1) != 0))) {
        phi_r8 = 4;
    }
    if ((((arg0 != 0) || (arg1 != 0)) && (arg2 != 0)) || ((arg3 != 0) && ((arg0 + 1) != 0)) || ((arg1 + 1) != 0) || ((arg2 + 1) != 0)) {
        phi_r8 = 5;
    }
    if ((((arg0 != 0) && (arg1 != 0)) || ((arg2 != 0) && (arg3 != 0))) && (((arg0 + 1) != 0) || ((arg1 + 1) != 0))) {
        phi_r8 = 6;
    }
    if (arg0 != 0) {
        if (arg1 != 0) {
            phi_r7 = arg2;
        } else {
            phi_r7 = arg3;
        }
        if (((s32) (arg0 + 1) == phi_r7) && ((arg1 + 1) != 0)) {
            phi_r8 = 7;
        }
    }
    if (arg0 == 0) {
        if (arg1 != 0) {

        } else {
            phi_r5 = arg3;
        }
        if (((s32) (arg0 + 1) == phi_r5) || ((arg1 + 1) != 0)) {
            goto block_53;
        }
    } else {
block_53:
        phi_r8 = 8;
    }
    return phi_r8;
}
