void test(s32 arg0, s32 arg1) {
    s32 var_eax;

    if (arg0 > 0) {
        var_eax = 0;
        do {
            *(arg1 + (var_eax * 4)) = var_eax;
            var_eax += 1;
        } while (var_eax < arg0);
    }
}
