extern s32 glob;

s32 test(s32 arg0) {
    s32 phi_r3;

    phi_r3 = arg0;
    if (phi_r3 != 4) {
        if (phi_r3 < 4) {
            if (phi_r3 != 2) {
                if (phi_r3 < 2) {
                    if (phi_r3 < 1) {
                        goto block_14;
                    }
                    return phi_r3 * phi_r3;
                }
                /* Duplicate return node #11. Try simplifying control flow for better match */
                return phi_r3 * 2;
            }
            phi_r3 -= 1;
            return phi_r3 * 2;
        }
        if (phi_r3 < 8) {
            if (phi_r3 < 6) {
                goto block_14;
            }
            glob = phi_r3 * 2;
            return 2;
        }
block_14:
        glob = phi_r3 / 2;
        return 2;
    }
    glob = phi_r3 + 1;
    return 2;
}
