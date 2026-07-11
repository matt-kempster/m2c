s32 other_function();                               /* extern */

s32 test(s32 arg0) {
    if (arg0 != 0) {
        return other_function();
    }
    return 0;
}
