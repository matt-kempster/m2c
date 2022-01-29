extern s32 glob;

s32 test(s32 arg0) {
    s32 phi_r3;

    phi_r3 = arg0;
    if (arg0 != 4) {
        if (arg0 < 4) {
            if (arg0 != 2) {
                if (arg0 < 2) {
                    if (arg0 < 1) {
                        goto block_14;
                    }
                    return arg0 * arg0;
                }
                /* Duplicate return node #11. Try simplifying control flow for better match */
                return phi_r3 * 2;
            }
            phi_r3 = arg0 - 1;
            return phi_r3 * 2;
        }
        if (arg0 < 8) {
            if (arg0 < 6) {
                goto block_14;
            }
            glob = arg0 * 2;
            return 2;
        }
block_14:
        glob = MIPS2C_ERROR(unknown instruction: addze $r3, $r3);
        return 2;
    }
    glob = arg0 + 1;
    return 2;
}
