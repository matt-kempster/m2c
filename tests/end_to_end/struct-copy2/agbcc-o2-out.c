extern ? d7;
extern ? s7;

void test(void) {

}

void test_0(void *arg0, void *arg1) {
    arg0->unk0 = (s32) arg1->unk0;
    arg0->unk4 = (s32) arg1->unk4;
}

s32 test_1(void *arg0, void *arg1) {
    arg0->unk0 = (s32) arg1->unk0;
    arg0->unk4 = (s32) arg1->unk4;
    arg0->unk8 = (s32) arg1->unk8;
    return arg0 + 0xC;
}

s32 test_2(void *arg0, void *arg1) {
    arg0->unk0 = (s32) arg1->unk0;
    arg0->unk4 = (s32) arg1->unk4;
    arg0->unk8 = (s32) arg1->unk8;
    return arg0 + 0xC;
}

s32 test_3(void *arg0, void *arg1) {
    arg0->unk0 = (s32) arg1->unk0;
    arg0->unk4 = (s32) arg1->unk4;
    arg0->unk8 = (s32) arg1->unk8;
    return arg0 + 0xC;
}

s32 test_4(void *arg0, void *arg1) {
    arg0->unk0 = (s32) arg1->unk0;
    arg0->unk4 = (s32) arg1->unk4;
    arg0->unk8 = (s32) arg1->unk8;
    return arg0 + 0xC;
}

void test_5(void *arg0, void *arg1) {
    arg0->unk0 = (s32) arg1->unk0;
    arg0->unk4 = (s32) arg1->unk4;
    arg0->unk8 = (s32) arg1->unk8;
    *(arg0 + 0xC) = *(arg1 + 0xC);
}

void test_6(void *arg0, void *arg1) {
    arg0->unk0 = (s32) arg1->unk0;
    arg0->unk4 = (s32) arg1->unk4;
    arg0->unk8 = (s32) arg1->unk8;
    *(arg0 + 0xC) = *(arg1 + 0xC);
}

void test_7(void *arg0, void *arg1) {
    arg0->unk0 = (s32) arg1->unk0;
    arg0->unk4 = (s32) arg1->unk4;
    arg0->unk8 = (s32) arg1->unk8;
    *(arg0 + 0xC) = *(arg1 + 0xC);
    d7.unk0 = (s32) s7.unk0;
    d7.unk4 = (s32) s7.unk4;
    d7.unk8 = (s32) s7.unk8;
    *(&d7 + 0xC) = *(&s7 + 0xC);
}
