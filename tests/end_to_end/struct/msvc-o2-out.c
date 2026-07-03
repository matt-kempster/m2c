void test(void *arg0, void *arg1) {
    s32 temp_ecx;

    temp_ecx = arg0->unk0;
    arg0->unk4 = (s32) (arg0->unk4 + temp_ecx);
    arg1->unk0 = temp_ecx;
    arg1->unk4 = (s32) arg0->unk4;
}
