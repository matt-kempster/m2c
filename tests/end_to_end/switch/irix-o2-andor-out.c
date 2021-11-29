extern s32 D_410150;

s32 test(s32 arg0) {
    s32 phi_a0;
    s32 phi_a0_2;

    phi_a0 = arg0;
    switch (arg0) {
    case 1:
        return arg0 * arg0;
    case 2:
        phi_a0 = arg0 - 1;
        /* fallthrough */
    case 3:
        return phi_a0 * 2;
    case 4:
        phi_a0_2 = arg0 + 1;
        D_410150 = phi_a0_2;
        return 2;
    case 6:
    case 7:
        phi_a0_2 = arg0 * 2;
        /* Duplicate return node #8. Try simplifying control flow for better match */
        D_410150 = phi_a0_2;
        return 2;
    default:
        phi_a0_2 = arg0 / 2;
        /* Duplicate return node #8. Try simplifying control flow for better match */
        D_410150 = phi_a0_2;
        return 2;
    }
}
