extern s32 D_410170;

s32 test(s32 arg0) {
    s32 phi_a0;
    s32 phi_a0_2;

    phi_a0_2 = arg0;
    if ((u32) (arg0 - 1) < 7U) {
        switch (arg0) {
        case 1:
            return arg0 * arg0;
        case 2:
            phi_a0_2 = arg0 - 1;
            // fallthrough
        case 3:
            return phi_a0_2 * 2;
        case 4:
            phi_a0 = arg0 + 1;
            // Duplicate return node #8. Try simplifying control flow for better match
            D_410170 = phi_a0;
            return 2;
        case 6:
        case 7:
            phi_a0 = arg0 * 2;
            // Duplicate return node #8. Try simplifying control flow for better match
            D_410170 = phi_a0;
            return 2;
        }
    } else {
    default:
        phi_a0 = arg0 / 2;
        D_410170 = phi_a0;
        return 2;
    }
}
