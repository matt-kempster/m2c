void *test(void *arg0) {
    s32 sp4;

    sp4 = arg0->unk0 + arg0->unk4;
    arg0->unk4 = sp4;
    return arg0;
}
