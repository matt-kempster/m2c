s32 test(s32 arg0) {
    s32 temp_v0;
    s32 var_t6;

    temp_v0 = arg0 + 0xFFC00000;
    var_t6 = temp_v0 >> 0x1F;
    if (temp_v0 < 0) {
        var_t6 = (s32) (temp_v0 + 0x7FFFFFFF) >> 0x1F;
    }
    return var_t6;
}
