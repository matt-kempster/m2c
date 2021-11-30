extern s32 D_410210;

s32 test(s32 arg0) {
    s32 phi_a0;
    s32 phi_a0_2;
    s32 phi_a0_3;
    s32 phi_a0_4;
    s32 phi_a0_5;

    phi_a0_2 = arg0;
    phi_a0_3 = arg0;
    phi_a0_4 = arg0;
    phi_a0_5 = arg0;
    switch (arg0) { // irregular
    case 1:
        return arg0 * arg0;
    case 2:
        phi_a0_2 = arg0 - 1;
        // fallthrough
    case 3:
    case 101:
    case 200:
        return (phi_a0_2 + 1) ^ phi_a0_2;
    case 107:
        phi_a0 = arg0 + 1;
        D_410210 = phi_a0;
        return 2;
    case -50:
        phi_a0 = arg0 - 1;
        // Duplicate return node #24. Try simplifying control flow for better match
        D_410210 = phi_a0;
        return 2;
    case 50:
        phi_a0 = arg0 + 1;
        // Duplicate return node #24. Try simplifying control flow for better match
        D_410210 = phi_a0;
        return 2;
    case 6:
    case 7:
        phi_a0_3 = arg0 * 2;
        // fallthrough
    case 102:
        phi_a0 = phi_a0_3;
        phi_a0_5 = phi_a0_3;
        if (D_410210 == 0) {
        case 103:
        case 104:
        case 105:
        case 106:
            phi_a0_4 = phi_a0_5 - 1;
        default:
            phi_a0 = phi_a0_4 / 2;
        }
        // Duplicate return node #24. Try simplifying control flow for better match
        D_410210 = phi_a0;
        return 2;
    }
}
