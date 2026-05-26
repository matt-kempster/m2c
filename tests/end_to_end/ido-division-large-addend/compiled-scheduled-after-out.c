s32 test(void *arg0) {
    return (arg0->unk4 - ((s32) (arg0->unk0 - 0x400000) / 32)) + arg0->unk8;
}
