s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 spC;
    s32 sp8;
    s32 sp4;
    s32 phi_t0;

    spC = arg0 + arg1;
    sp8 = arg1 + arg2;
    sp4 = 0;
    if ((spC != 0) || ((sp8 != 0) && ((arg0 * arg1) != 0)) || ((arg3 != 0) && (arg0 != 0))) {
        sp4 = 1;
    }
    if ((arg0 != 0) && ((arg1 != 0) || (arg2 != 0)) && ((arg3 != 0) || ((arg0 + 1) != 0))) {
        sp4 = 2;
    }
    if (((arg0 != 0) && (arg3 != 0)) || (((arg1 != 0) || (arg2 != 0)) && ((arg0 + 1) != 0))) {
        sp4 = 3;
    }
    if ((arg0 != 0) && (arg1 != 0) && ((arg2 != 0) || (arg3 != 0)) && (((arg0 + 1) != 0) || ((arg1 + 1) != 0))) {
        sp4 = 4;
    }
    if ((((arg0 != 0) || (arg1 != 0)) && (arg2 != 0)) || ((arg3 != 0) && ((arg0 + 1) != 0)) || ((arg1 + 1) != 0) || ((arg2 + 1) != 0)) {
        sp4 = 5;
    }
    if (((arg0 == 0) || (arg1 == 0)) && ((arg2 != 0) && (arg3 != 0) && (((arg0 + 1) != 0) || ((arg1 + 1) != 0)))) {
        sp4 = 6;
    }
    if (arg0 != 0) {
        if (arg1 != 0) {
            phi_t0 = arg2;
        } else {
            phi_t0 = arg3;
        }
        if (phi_t0 == (arg0 + 1)) {
            sp4 = 7;
        }
    }
    return sp4;
}
