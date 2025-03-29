static ? s7;
static ? d7;

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

void test_7(Test7 *a, Test7 *b) {
        *a = s7;                                        /* size 0xF */
        d7 = *b;                                        /* size 0xF */
}
