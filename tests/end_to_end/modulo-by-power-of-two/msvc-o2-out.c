void test(s32 *arg0) {
    s32 var_ecx;

    var_ecx = *arg0 & ~0x7FFFFFFE;
    if (var_ecx < 0) {
        var_ecx = ((var_ecx - 1) | ~1) + 1;
    }
    *arg0 = var_ecx;
}
