void test(s32 arg0, s32 arg1, s32 *arg2) {
    if (arg0 != 0) {
        *arg2 = arg0;
    }
    if (arg1 != 0) {
        *arg2 = arg1;
    }
}

void test_bf(s32 arg0, s32 arg1, s32 *arg2) {
    if (arg0 == 0) {
        *arg2 = arg0;
    }
    if (arg1 == 0) {
        *arg2 = arg1;
    }
}
