void *test(void *a0) {
    s32 sp4;

    sp4 = (s32) (*a0 + a0->unk4);
    a0->unk4 = sp4;
    return;
    // (possible return value: a0)
}
