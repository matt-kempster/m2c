s32 test(u32 arg0, u32 arg1) {
    if (arg0 <= arg1) {
        if (arg0 >= arg1) {
            return 0;
        }
        /* Duplicate return node #5. Try simplifying control flow for better match */
        return 2;
    }
    if (arg0 > 0x100U) {
        return 1;
    }
    return 2;
}
