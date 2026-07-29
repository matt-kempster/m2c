s32 test(void) {
    return -1;
}

s32 test_min(void) {
    return -0x80;
}

s32 test_max(void) {
    return 0x7F;
}

s32 test_add(s32 arg0) {
    return arg0 - 1;
}

s32 test_compare(s32 arg0) {
    return arg0 == -1;
}
