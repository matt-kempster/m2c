s32 test(s32 arg0) {
    s32 var_r1;
    s32 var_r2;
    s32 var_r3;
    s32 var_r5;

    var_r3 = 0;
    var_r5 = 0;
    if (arg0 > 0) {
        do {
            if (arg0 > 0) {
                var_r2 = 0;
                var_r1 = arg0;
                do {
                    var_r5 += var_r2;
                    var_r2 += var_r3;
                    var_r1 -= 1;
                } while (var_r1 != 0);
            }
            var_r3 += 1;
        } while (var_r3 < arg0);
    }
    return var_r5;
}
