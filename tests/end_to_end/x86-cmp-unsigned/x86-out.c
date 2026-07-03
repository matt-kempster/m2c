s32 test(u32 arg0, u32 arg1) {
    if (arg0 <= arg1) {
        if (arg0 >= arg1) {
            return 0;
        }
        return 2;
    }
    if (arg0 > 0x100U) {
        return 1;
    }
    return 2;
}
