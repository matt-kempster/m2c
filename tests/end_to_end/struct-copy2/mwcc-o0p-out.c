static ? d7;
static ? s7;

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

void test_7(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(arg0, arg1, 0xF);
    M2C_STRUCT_COPY(&d7, &s7, 0xF);
}
