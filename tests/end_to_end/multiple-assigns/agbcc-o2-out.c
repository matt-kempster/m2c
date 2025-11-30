extern s32 glob;

s32 test(s32 arg0, s32 arg2) {
    s32 temp_r0;
    s32 temp_r0_2;
    s32 temp_r0_3;
    s32 temp_r0_4;
    s32 temp_r0_5;
    s32 var_r0;
    s32 var_r2;

    var_r0 = arg0;
    var_r2 = arg2;
    if (var_r0 == 5) {
        do {
            glob = var_r0;
            temp_r0 = var_r0 + 1;
            glob = temp_r0;
            temp_r0_2 = temp_r0 + 1;
            glob = temp_r0_2;
            temp_r0_3 = temp_r0_2 + 1;
            glob = temp_r0_3;
            var_r2 = temp_r0_3;
            temp_r0_4 = var_r2 + 1;
            glob = temp_r0_4;
            glob = temp_r0_4;
            temp_r0_5 = temp_r0_4 + 1;
            glob = temp_r0_5;
            var_r0 = temp_r0_5 + 1;
            glob = var_r2;
        } while (var_r0 == 5);
    }
    return var_r2;
}
