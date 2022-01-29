void test(s32 arg0) {
    if (arg0 < 5) {
        if (arg0 < 1) {

        } else {
            *NULL = 1;
            if (arg0 == 1) {
                *NULL = 2;
            }
        }
    } else if (arg0 < 7) {
        if (arg0 == 1) {
            *NULL = 1;
        } else {
            *NULL = 2;
        }
    }
    if (arg0 < 5) {
        if (arg0 < 1) {
            return;
        }
        *NULL = 1;
        if (arg0 == 1) {
            *NULL = 2;
        }
    } else if (arg0 < 7) {
        if (arg0 == 1) {
            *NULL = 1;
            return;
        }
        *NULL = 2;
    }
}
