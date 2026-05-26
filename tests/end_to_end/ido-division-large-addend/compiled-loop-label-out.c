s32 test(s32 *arg0, s32 arg1) {
    s32 var_v0;
    s32 var_v1;

    var_v1 = 0;
    var_v0 = 0;
    if (arg1 > 0) {
        do {
            var_v0 += 1;
            var_v1 += (s32) (*arg0 - 0x400000) / 32;
        } while (arg1 != var_v0);
    }
    return var_v1;
}
