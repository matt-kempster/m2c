extern s32 D_4101D0;

void test(s32 arg0) {
    u32 temp_t1;
    u32 temp_t6;

    temp_t6 = arg0 - 1;
    if (temp_t6 < 6U) {
        goto **(&jtbl_4001A0 + (temp_t6 * 4)); // switch 1
    case 0: // switch 1
        D_4101D0 = 1;
        if (arg0 == 1) {
            D_4101D0 = 2;
        }
        goto block_7;
    case 1: // switch 1
        if (arg0 == 1) {
            D_4101D0 = 1;
        } else {
            D_4101D0 = 2;
        }
        goto block_7;
    }
block_7:
    temp_t1 = arg0 - 1;
    if (temp_t1 < 6U) {
        goto **(&jtbl_4001B8 + (temp_t1 * 4)); // switch 2
    case 0: // switch 2
        D_4101D0 = 1;
        if (arg0 == 1) {
            D_4101D0 = 2;
            return;
        }
        return;
    case 1: // switch 2
        if (arg0 == 1) {
            D_4101D0 = 1;
            return;
        }
        D_4101D0 = 2;
        // Duplicate return node #14. Try simplifying control flow for better match
        return;
    } else {
        // Duplicate return node #14. Try simplifying control flow for better match
    }
}
