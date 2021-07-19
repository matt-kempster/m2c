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
    if (arg0 >= 0x33) {
        if (arg0 >= 0x6C) {
            if (arg0 != 0xC8) {
                goto block_23;
            }
            // Duplicate return node #16. Try simplifying control flow for better match
            return (phi_a0_2 + 1) ^ phi_a0_2;
        }
        switch (arg0) { // switch 1
        case 107: // switch 1
            phi_a0 = arg0 + 1;
            // Duplicate return node #24. Try simplifying control flow for better match
            D_410210 = phi_a0;
            return 2;
        }
    } else {
        if (arg0 >= 8) {
            if (arg0 != 0x32) {
                goto block_23;
            }
            phi_a0 = arg0 + 1;
            // Duplicate return node #24. Try simplifying control flow for better match
            D_410210 = phi_a0;
            return 2;
        }
        if (arg0 >= -0x31) {
            switch (arg0) { // switch 2
            case 1: // switch 2
                return arg0 * arg0;
            case 2: // switch 2
                phi_a0_2 = arg0 - 1;
                // fallthrough
            case 3: // switch 2
            case 101: // switch 1
                return (phi_a0_2 + 1) ^ phi_a0_2;
            case 6: // switch 2
            case 7: // switch 2
                phi_a0_3 = arg0 * 2;
            case 102: // switch 1
                phi_a0 = phi_a0_3;
                phi_a0_5 = phi_a0_3;
                if (D_410210 == 0) {
                case 103: // switch 1
                case 104: // switch 1
                case 105: // switch 1
                case 106: // switch 1
                    phi_a0_4 = phi_a0_5 - 1;
                    goto block_23;
                }
                // Duplicate return node #24. Try simplifying control flow for better match
                D_410210 = phi_a0;
                return 2;
            }
        } else {
            if (arg0 != -0x32) {
            default: // switch 2
            default: // switch 1
block_23:
                phi_a0 = phi_a0_4 / 2;
            } else {
                phi_a0 = arg0 - 1;
            }
            D_410210 = phi_a0;
            return 2;
        }
    }
}
