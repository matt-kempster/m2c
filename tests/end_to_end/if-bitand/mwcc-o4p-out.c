extern s32 glob;

s32 test(s32 arg0) {
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
        return arg0;
    }
    return arg0;
}
