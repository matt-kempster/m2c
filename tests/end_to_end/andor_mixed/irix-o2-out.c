? test(s32 arg0, s32 arg1, s32 arg2, s32 arg3); // static

? test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    ? phi_v1;

    if (((arg0 + arg1) != 0) || (((arg1 + arg2) != 0) && ((arg0 * arg1) != 0)) || ((phi_v1 = 0, (arg3 != 0)) && (phi_v1 = 0, (arg0 != 0)))) {
        phi_v1 = 1;
    }
    if ((arg0 != 0) && ((arg1 != 0) || (arg2 != 0)) && ((arg3 != 0) || ((arg0 + 1) != 0))) {
        phi_v1 = 2;
    }
    if (((arg0 != 0) && (arg3 != 0)) || (((arg1 != 0) || (arg2 != 0)) && ((arg0 + 1) != 0))) {
        phi_v1 = 3;
    }
    if ((arg0 != 0) && (arg1 != 0) && ((arg2 != 0) || (arg3 != 0)) && (((arg0 + 1) != 0) || ((arg1 + 1) != 0))) {
        phi_v1 = 4;
    }
    if ((((arg0 != 0) || (arg1 != 0)) && (arg2 != 0)) || ((arg3 != 0) && ((arg0 + 1) != 0)) || ((arg1 + 1) != 0) || ((arg2 + 1) != 0)) {
        phi_v1 = 5;
    }
    return phi_v1;
}
