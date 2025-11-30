extern s32 glob;

void test(s32 arg0) {
    glob = *(arg0 + 0x12348);
}
