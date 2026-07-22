s8 test(s8 *ptr) {
    return *ptr;
}

s16 test_loadw(s16 *ptr) {
    return *ptr;
}

s32 test_loadl(s32 *ptr) {
    return *ptr;
}

void test_storeb(s8 *ptr, s8 value) {
    *ptr = value;
}

void test_storew(s16 *ptr, s16 value) {
    *ptr = value;
}

void test_storel(s32 *ptr, s32 value) {
    *ptr = value;
}

s8 test_loadb_postinc(s8 **ptr) {
    s8 *temp_r1;

    temp_r1 = *ptr;
    *ptr = temp_r1 + 1;
    return *temp_r1;
}

s16 test_loadw_postinc(s16 **ptr) {
    s16 *temp_r1;

    temp_r1 = *ptr;
    *ptr = temp_r1 + 2;
    return *temp_r1;
}

s32 test_loadl_postinc(s32 **ptr) {
    s32 *temp_r1;

    temp_r1 = *ptr;
    *ptr = temp_r1 + 4;
    return *temp_r1;
}

void test_storeb_predec(s8 **ptr, s8 value) {
    s8 *temp_r1;

    temp_r1 = *ptr - 1;
    *temp_r1 = value;
    *ptr = temp_r1;
}

void test_storew_predec(s16 **ptr, s16 value) {
    s16 *temp_r1;

    temp_r1 = *ptr - 2;
    *temp_r1 = value;
    *ptr = temp_r1;
}

void test_storel_predec(s32 **ptr, s32 value) {
    s32 *temp_r1;

    temp_r1 = *ptr - 4;
    *temp_r1 = value;
    *ptr = temp_r1;
}
