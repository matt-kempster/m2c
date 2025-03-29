static ? d7;
static ? s7;

void test(void) {

}

void test_0(? *arg0, ? *arg1) {
        *arg0 = *arg1;                                  /* size 0x8 */
}

void test_1(? *arg0, ? *arg1) {
        *arg0 = *arg1;                                  /* size 0x9 */
}

void test_2(? *arg0, ? *arg1) {
        *arg0 = *arg1;                                  /* size 0xA */
}

void test_3(? *arg0, ? *arg1) {
        *arg0 = *arg1;                                  /* size 0xB */
}

void test_4(? *arg0, ? *arg1) {
        *arg0 = *arg1;                                  /* size 0xC */
}

void test_5(? *arg0, ? *arg1) {
        *arg0 = *arg1;                                  /* size 0xD */
}

void test_6(? *arg0, ? *arg1) {
        *arg0 = *arg1;                                  /* size 0xE */
}

void test_7(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(arg0, arg1, 0xF);
    M2C_STRUCT_COPY(&d7, &s7, 0xF);
}
