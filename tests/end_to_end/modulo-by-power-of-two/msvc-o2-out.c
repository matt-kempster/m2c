void test(s32 *ptr) {
    s32 var_ecx;

    var_ecx = *ptr & ~0x7FFFFFFE;
    if (var_ecx < 0) {
        var_ecx = ((var_ecx - 1) | ~1) + 1;
    }
    *ptr = var_ecx;
}
