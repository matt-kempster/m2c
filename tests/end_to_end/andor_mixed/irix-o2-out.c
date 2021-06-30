s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 phi_v1;

    if (((arg0 + arg1) != 0) || (((arg1 + arg2) != 0) && ((arg0 * arg1) != 0)) || ((phi_v1 = 0, (arg3 != 0)) && (phi_v1 = 0, (arg0 != 0)))) {
        phi_v1 = arg0 + arg3;
    }
    if ((arg0 != 0) && ((arg1 != 0) || (arg2 != 0)) && ((arg3 != 0) || ((arg0 + 1) != 0))) {
        phi_v1 = arg1 + arg3;
    }
    if (((arg0 != 0) && (arg3 != 0)) || (((arg1 != 0) || (arg2 != 0)) && ((arg3 + arg2) != 0))) {
        phi_v1 = arg2 + arg3;
    }
    return phi_v1;
}
