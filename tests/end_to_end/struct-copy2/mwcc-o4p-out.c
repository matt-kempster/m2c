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
