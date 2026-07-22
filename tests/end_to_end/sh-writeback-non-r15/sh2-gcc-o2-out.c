s32 test(s32 **ptr) {
    s32 *temp_r1;

    temp_r1 = *ptr;
    *ptr = temp_r1 + 4;
    return *temp_r1;
}

void test_storel_predec(s32 **ptr, s32 value) {
    s32 *temp_r1;

    temp_r1 = *ptr - 4;
    *temp_r1 = value;
    *ptr = temp_r1;
}

s8 test_loadb_predec(s8 **ptr) {
    s8 *temp_r1;

    temp_r1 = *ptr - 1;
    *ptr = temp_r1;
    return *temp_r1;
}

s16 test_loadw_predec(s16 **ptr) {
    s16 *temp_r1;

    temp_r1 = *ptr - 2;
    *ptr = temp_r1;
    return *temp_r1;
}
