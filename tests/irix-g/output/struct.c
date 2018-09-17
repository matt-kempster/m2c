void *test(void *arg0) {
    s32 sp4;

    sp4 = (s32) (*arg0 + arg0->unk4);
    arg0->unk4 = sp4;
    return;
    // (possible return value: arg0)
}
