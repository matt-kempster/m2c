s32 func_00400090(s32);                             /* static */

s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 sp24;
    s32 sp20;
    s32 temp_t3_60;
    s32 temp_t5_82;
    s32 temp_t7_13;
    s32 temp_v0_22;
    s32 temp_v0_47;
    s32 var_a2;
    s32 var_s0;
    s32 var_t0;
    s32 var_v1;
    s32 var_v1_2;
    s32 var_v1_3;
    s32 var_v1_4;

    var_a2 = arg2;
    var_s0 = arg0 + arg1;
    temp_t7_13 = arg1 + var_a2;
    sp20 = temp_t7_13;
    var_t0 = temp_t7_13;
    if ((var_s0 != 0) || (temp_t7_13 != 0) || (temp_v0_22 = func_00400090(temp_t7_13), var_a2 = arg2, var_t0 = temp_v0_22, (temp_v0_22 != 0)) || (var_s0 = 2, (arg3 != 0))) {
        var_v1_2 = 1;
    } else {
        var_v1_2 = -2;
        if (arg0 != 0) {
            var_v1_2 = -1;
        }
    }
    var_v1_3 = var_v1_2 + var_a2;
    if ((var_s0 != 0) && (var_t0 != 0)) {
        var_s0 += var_t0;
        sp24 = var_v1_3;
        temp_v0_47 = func_00400090(var_s0);
        var_t0 = temp_v0_47;
        if ((temp_v0_47 != 0) && (arg3 != 0)) {
            if (var_v1_3 < 5) {
                do {
                    temp_t3_60 = (var_v1_3 + 1) * 2;
                    var_v1_3 = temp_t3_60;
                } while (temp_t3_60 < 5);
            }
            var_v1_3 += 5;
        }
    }
    if ((var_s0 != 0) && (var_t0 != 0) && (sp24 = var_v1_3, var_v1_4 = var_v1_3, (func_00400090(var_s0 + var_t0) != 0)) && (arg3 != 0)) {
        if (var_v1_4 < 5) {
            do {
                temp_t5_82 = (var_v1_4 + 1) * 2;
                var_v1_4 = temp_t5_82;
            } while (temp_t5_82 < 5);
        }
        var_v1 = var_v1_4 + 5;
    } else {
        var_v1 = var_v1_3 + 6;
    }
    return var_v1;
}
