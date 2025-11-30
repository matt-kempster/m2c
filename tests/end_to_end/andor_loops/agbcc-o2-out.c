s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 var_r1;
    s32 var_r3;

    var_r1 = 0;
loop_2:
    if (arg0 != 0) {
        if ((arg1 == 0) && (arg2 == 0)) {
            if (arg0 == 0) {
                goto block_6;
            }
            goto loop_8;
        }
        var_r1 += 1;
        goto loop_2;
    }
block_6:
    if ((arg1 != 0) && (arg2 != 0)) {
loop_8:
        var_r1 += 1;
        if (arg0 != 0) {
            goto loop_8;
        }
        if (arg1 != 0) {
            if (arg2 == 0) {

            } else {
                goto loop_8;
            }
        }
    }
loop_14:
    if (arg0 != 0) {
        var_r1 += 1;
        if (((arg1 == 0) || ((arg2 == 0) && (arg3 == 0))) && (var_r1 += 1, (arg1 == 0)) && ((arg2 == 0) || (arg3 == 0))) {
            var_r1 += 2;
            if (arg2 != 0) {
                if (arg3 == 0) {
                    goto block_13;
                }
            } else {
block_13:
                var_r1 += 1;
                goto loop_14;
            }
        } else {
            goto loop_14;
        }
    }
    var_r3 = 0;
loop_25:
    if ((arg0 != 0) || (arg1 != 0)) {
        var_r1 += 1;
        var_r3 += arg2 + arg3;
        if (var_r3 <= 9) {
            goto loop_25;
        }
    }
    return var_r1;
}
