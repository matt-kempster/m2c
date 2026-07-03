extern s32 _glob;

void test(void *arg0, s32 arg1) {
    _glob = *(arg0 + ((arg1 * 4) + 4));
    _glob = arg0 + ((arg1 * 4) + 4);
    _glob = *(arg0 + ((arg1 * 8) + 0x30));
    _glob = arg0 + ((arg1 * 8) + 0x30);
    _glob = *((arg1 << 7) + (arg0 + 0x7C));
    _glob = arg0->unk48;
    _glob = arg0 + 0x48;
}
