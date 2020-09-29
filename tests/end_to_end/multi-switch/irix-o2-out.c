s32 test(s32 arg0) {
    u32 temp_t6;
    u32 temp_t7;
    s32 phi_a0;
    s32 phi_a0_2;
    s32 phi_a0_3;
    s32 phi_a0_4;
    s32 phi_a0_5;

    if (arg0 >= 0x33) {
        temp_t6 = arg0 - 0x65;
        if (arg0 >= 0x6C) {
            phi_a0_2 = arg0;
            if (arg0 == 0xC8) {
            case 0: // switch 1
            case 2: // switch 2
                return (phi_a0_2 + 1) ^ phi_a0_2;
            case 6: // switch 1
block_17:
                phi_a0 = arg0 + 1;
block_24:
                D_410210 = phi_a0;
                return 2;
            }
            phi_a0_3 = arg0;
        default: // switch 2
block_23:
            phi_a0 = phi_a0_3 / 2;
        } else {
            phi_a0_3 = arg0;
            if (temp_t6 < 7U) {
                phi_a0_2 = arg0;
                phi_a0_4 = arg0;
                phi_a0_5 = arg0;
                goto **(&jtbl_4001D0 + (temp_t6 * 4)); // switch 1
block_6:
                if (arg0 >= 8) {
                    if (arg0 != 0x32) {
                        phi_a0_3 = arg0;
                        goto block_23;
                    } else {
                        phi_a0 = arg0 + 1;
                        goto block_24;
                    case 5: // switch 2
                    case 6: // switch 2
                        phi_a0_4 = arg0 * 2;
                    case 1: // switch 1
                        phi_a0 = phi_a0_4;
                        phi_a0_5 = phi_a0_4;
                        if (D_410210 == 0) {
                        default: // switch 1
                            phi_a0_3 = phi_a0_5 - 1;
                            goto block_23;
                        }
                    }
                } else {
                    temp_t7 = arg0 - 1;
                    if (arg0 >= -0x31) {
                        phi_a0_3 = arg0;
                        if (temp_t7 < 7U) {
                            phi_a0_2 = arg0;
                            phi_a0_3 = arg0;
                            goto **(&jtbl_4001EC + (temp_t7 * 4)); // switch 2
block_12:
                            if (arg0 != -0x32) {
                                phi_a0_3 = arg0;
                                goto block_23;
                            case 0: // switch 2
                                return arg0 * arg0;
                            case 1: // switch 2
                                phi_a0_2 = arg0 - 1;
                            case 0: // switch 1
                            case 2: // switch 2
                                return (phi_a0_2 + 1) ^ phi_a0_2;
                                goto block_17;
                            } else {
                                phi_a0 = arg0 - 1;
                            }
                        } else {
                            goto block_23;
                        }
                    } else {
                        goto block_12;
                    }
                }
            } else {
                goto block_23;
            }
        }
    } else {
        goto block_6;
    }
block_24:
    D_410210 = phi_a0;
    return 2;
}
