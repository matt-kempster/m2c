s32 foo(s32);                                       /* static */

s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 var_r0;
    s32 var_r4;
    s32 var_r4_2;
    s32 var_r4_3;
    s32 var_r5;

    var_r5 = arg0 + arg1;
    var_r0 = arg1 + arg2;
    if ((var_r5 != 0) || (var_r0 != 0) || (var_r0 = foo(0), (var_r0 != 0)) || (var_r5 = 2, (arg3 != 0))) {
        var_r4_2 = 1;
    } else {
        var_r4_2 = -2;
        if (arg0 != 0) {
            var_r4_2 = -1;
        }
    }
    var_r4_3 = var_r4_2 + arg2;
    if (var_r5 != 0) {
        if (var_r0 != 0) {
            var_r5 += var_r0;
            var_r0 = foo(var_r5);
            if ((var_r0 != 0) && (arg3 != 0)) {
                if (var_r4_3 <= 4) {
                    do {
                        var_r4_3 = (var_r4_3 + 1) * 2;
                    } while (var_r4_3 <= 4);
                }
                var_r4_3 += 5;
            }
        }
        if ((var_r5 != 0) && (var_r0 != 0) && (foo(var_r5 + var_r0) != 0) && (arg3 != 0)) {
            if (var_r4_3 <= 4) {
                do {
                    var_r4_3 = (var_r4_3 + 1) * 2;
                } while (var_r4_3 <= 4);
            }
            var_r4 = var_r4_3 + 5;
        } else {
            goto block_21;
        }
    } else {
block_21:
        var_r4 = var_r4_3 + 6;
    }
    return var_r4;
}
