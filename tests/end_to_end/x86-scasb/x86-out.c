s32 test(s32 arg0) {
    return ~(-1 - ((M2C_MEMCHR(arg0, 0x3BU, -1) - arg0) + 1)) - 1;
}
