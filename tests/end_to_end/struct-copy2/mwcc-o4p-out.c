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

void test_7(void *arg0, void *arg1) {
    arg0->unk0 = (s32) arg1->unk0;
    arg0->unk4 = (s32) arg1->unk4;
    arg0->unk8 = (s32) arg1->unk8;
    arg0->unkC = (u16) arg1->unkC;
    arg0->unkE = (u8) arg1->unkE;
    d7.unk0 = (s32) s7.unk0;
    d7.unk4 = (s32) s7.unk4;
    d7.unk8 = (s32) s7.unk8;
    d7.unkC = (u16) s7.unkC;
    d7.unkE = (u8) s7.unkE;
}
