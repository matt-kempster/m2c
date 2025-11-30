void test(s32 arg0, s32 arg1) {
    s32 var_r2;

    var_r2 = 0;
    if (arg1 > 0) {
        do {
            *(arg0 + var_r2) = 0;
            var_r2 += 1;
        } while (var_r2 < arg1);
    }
}
