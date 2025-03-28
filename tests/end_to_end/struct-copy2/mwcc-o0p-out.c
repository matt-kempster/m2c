static ? s7;
static ? d7;

void test(void) {

}

void test_0(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(arg0, arg1, 8);
}

void test_1(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(arg0, arg1, 9);
}

void test_2(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(arg0, arg1, 0xA);
}

void test_3(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(arg0, arg1, 0xB);
}

void test_4(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(arg0, arg1, 0xC);
}

void test_5(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(arg0, arg1, 0xD);
}

void test_6(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(arg0, arg1, 0xE);
}

void test_7(Test7 *a, Test7 *b) {
    M2C_STRUCT_COPY(a, &s7, 0xF);
    M2C_STRUCT_COPY(&d7, b, 0xF);
}
