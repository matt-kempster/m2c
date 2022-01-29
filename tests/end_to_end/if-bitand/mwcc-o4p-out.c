s32 test(s32 arg0) {
    if ((*NULL & 1) != 0) {
        *NULL = 0;
    }
    if ((*NULL & 0x10000) != 0) {
        *NULL = 0;
    }
    if ((*NULL & 0x80000000) != 0) {
        *NULL = 0;
    }
    if ((*NULL & 1) != 0) {
        *NULL = 0;
    }
    if ((*NULL & 0x10000) != 0) {
        *NULL = 0;
    }
    if ((*NULL & 0x80000000) != 0) {
        *NULL = 0;
        return arg0;
    }
    return arg0;
}
