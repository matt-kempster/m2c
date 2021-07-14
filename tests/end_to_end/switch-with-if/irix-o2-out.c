extern s32 D_4101D0;

void test(s32 arg0) {
    if ((u32) (arg0 - 1) < 6U) {
        switch (arg0) { // switch 1
        case 1: // switch 1
            D_4101D0 = 1;
            if (arg0 == 1) {
                D_4101D0 = 2;
            }
            break;
        case 2: // switch 1
            if (arg0 == 1) {
                D_4101D0 = 1;
            } else {
                D_4101D0 = 2;
            }
            break;
        }
    }
    if ((u32) (arg0 - 1) < 6U) {
        switch (arg0) { // switch 2
        case 1: // switch 2
            D_4101D0 = 1;
            if (arg0 == 1) {
                D_4101D0 = 2;
                return;
            }
            // Duplicate return node #14. Try simplifying control flow for better match
            return;
        case 2: // switch 2
            if (arg0 == 1) {
                D_4101D0 = 1;
                return;
            }
            D_4101D0 = 2;
            // Duplicate return node #14. Try simplifying control flow for better match
            return;
        }
    } else {
    default: // switch 2
    }
}
