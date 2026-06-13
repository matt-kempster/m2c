extern s32 global;

void test(u32 arg0, u32 arg1) {
    s32 var_r2;
    s32 var_r2_2;

    var_r2 = 0;
    if (arg0 < arg1) {
        var_r2 = 1;
    }
    global = var_r2;
    var_r2_2 = 0;
    if (arg0 <= arg1) {
        var_r2_2 = 1;
    }
    global = var_r2_2;
}
