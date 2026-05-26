s32 test(s32 arg0) {
    s32 var_at;

    var_at = 0xFFC00000;
    if ((arg0 + 0xFFC00000) < 0) {
        var_at = arg0 + 0xFFC0001F;
    }
    return var_at;
}
