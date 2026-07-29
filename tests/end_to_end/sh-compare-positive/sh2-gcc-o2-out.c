s32 test(s32 arg0) {
    s32 var_r0;

    if (arg0 < 0) {
        return -1;
    }
    var_r0 = 1;
    if (arg0 <= 0) {
        var_r0 = 0;
    }
    return var_r0;
}
