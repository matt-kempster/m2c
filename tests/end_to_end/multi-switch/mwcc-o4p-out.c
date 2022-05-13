extern s32 glob;

s32 test(s32 arg0) {
    s32 phi_r3;

    phi_r3 = arg0;
    if (arg0 != 0x00000032) {
        if (arg0 < 0x32) {
            switch (arg0) { // switch 1; irregular
            case 1: // switch 1
                return arg0 * arg0;
            case 2: // switch 1
                phi_r3 = arg0 - 1;
                // Duplicate return node #23. Try simplifying control flow for better match
                return phi_r3 ^ (phi_r3 + 1);
            case -50: // switch 1
                glob = arg0 - 1;
                return 2;
            default: // switch 1
                phi_r3 = arg0 * 2;
                goto block_28;
            }
        } else {
            switch (arg0) { // irregular
            case 0xC8:
            case 0x65:
            case 3: // switch 1
                return phi_r3 ^ (phi_r3 + 1);
            case 0x6B:
                glob = arg0 + 1;
                return 2;
            case 0x66:
block_28:
                if ((s32) glob == 0) {
                    phi_r3 -= 1;
                    phi_r3 = phi_r3 / 2;
                }
                glob = phi_r3;
                return 2;
            }
        }
    } else {
        glob = arg0 + 1;
        return 2;
    }
}
