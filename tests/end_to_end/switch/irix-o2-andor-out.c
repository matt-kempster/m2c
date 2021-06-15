s32 test(s32 arg0) {
    u32 temp_t6;
    s32 phi_a0;
    s32 phi_a0_2;

    temp_t6 = arg0 - 1;
    if (temp_t6 < 7U) {
        phi_a0_2 = arg0;
        goto **(&jtbl_400130 + (temp_t6 * 4));
    case 0:
        return arg0 * arg0;
    case 1:
        phi_a0_2 = arg0 - 1;
    case 2:
        return phi_a0_2 * 2;
    case 3:
        phi_a0 = arg0 + 1;
        D_410150 = phi_a0;
        return 2;
    case 4:
block_7:
        phi_a0 = arg0 / 2;
        D_410150 = phi_a0;
        return 2;
    default:
        phi_a0 = arg0 * 2;
        D_410150 = phi_a0;
        return 2;
    } else {
        goto block_7;
    }
}
