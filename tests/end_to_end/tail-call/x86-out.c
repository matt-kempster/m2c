s32 _other_function();                              /* extern */

s32 test(s32 arg0) {
    if (arg0 != 0) {
        return _other_function();
    }
    return 0;
}
