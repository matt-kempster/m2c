extern s32 glob;

void test(void) {
    if ((glob & 1) != 0) {
        glob = 0;
    }
    if ((glob & 0x10000) != 0) {
        glob = 0;
    }
    if ((glob & 0x80000000) != 0) {
        glob = 0;
    }
    if ((glob & 1) != 0) {
        glob = 0;
    }
    if ((glob & 0x10000) != 0) {
        glob = 0;
    }
    if ((glob & 0x80000000) != 0) {
        glob = 0;
    }
}
