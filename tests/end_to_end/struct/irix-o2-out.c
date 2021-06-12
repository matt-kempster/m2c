void *test(void *arg0, void *arg1) {
    arg0->unk4 = (s32) (arg0->unk0 + arg0->unk4);
    arg1->unk0 = (s32) arg0->unk0;
    arg1->unk4 = (s32) arg0->unk4;
    return arg0;
}
