extern u32 global;

void test(s32 arg0, s32 arg1, s32 arg2) {
    s32 temp_r1;
    u32 var_r0;
    u32 var_r0_2;
    u32 var_r0_3;
    u32 var_r0_4;

    var_r0 = 0;
    if (arg0 == arg1) {
        var_r0 = 1;
    }
    global = var_r0;
    temp_r1 = arg0 ^ arg2;
    global = (u32) ((0 - temp_r1) | temp_r1) >> 0x1F;
    var_r0_2 = 0;
    if (arg0 < arg1) {
        var_r0_2 = 1;
    }
    global = var_r0_2;
    var_r0_3 = 0;
    if (arg0 <= arg1) {
        var_r0_3 = 1;
    }
    global = var_r0_3;
    var_r0_4 = 0;
    if (arg0 == 0) {
        var_r0_4 = 1;
    }
    global = var_r0_4;
    global = (u32) ((0 - arg1) | arg1) >> 0x1F;
}
