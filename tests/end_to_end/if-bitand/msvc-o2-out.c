extern s32 _glob;

void test(void) {
    if ((s8) _glob & 1) {
        _glob = 0;
    }
    if (_glob & 0x10000) {
        _glob = 0;
    }
    if (_glob & ~0x7FFFFFFF) {
        _glob = 0;
    }
    if (1 & (s8) _glob) {
        _glob = 0;
    }
    if (_glob & 0x10000) {
        _glob = 0;
    }
    if (_glob & ~0x7FFFFFFF) {
        _glob = 0;
    }
}
