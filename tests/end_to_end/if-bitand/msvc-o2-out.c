void test(void) {
    if ((s8) glob & 1) {
        glob = 0;
    }
    if (glob & 0x10000) {
        glob = 0;
    }
    if (glob & ~0x7FFFFFFF) {
        glob = 0;
    }
    if (1 & (s8) glob) {
        glob = 0;
    }
    if (glob & 0x10000) {
        glob = 0;
    }
    if (glob & ~0x7FFFFFFF) {
        glob = 0;
    }
}
