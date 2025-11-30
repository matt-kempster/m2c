s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 var_r2;
    s32 var_r6;

    var_r6 = 0;
    if (((arg0 + arg1) == 0) && (((arg1 + arg2) == 0) || ((arg0 * arg1) == 0))) {
        if (arg3 != 0) {
            var_r2 = arg0 + 1;
            if (arg0 != 0) {
                goto block_5;
            }
            goto block_14;
        }
        goto block_6;
    }
block_5:
    var_r6 = 1;
block_6:
    var_r2 = arg0 + 1;
    if (arg0 != 0) {
        if (((arg1 != 0) || (arg2 != 0)) && ((var_r2 = arg0 + 1, (arg3 != 0)) || (var_r2 != 0))) {
            var_r6 = 2;
        }
        if ((arg0 == 0) || (arg3 == 0)) {
            goto block_14;
        }
        goto block_17;
    }
block_14:
    if (((arg1 != 0) || (arg2 != 0)) && (var_r2 != 0)) {
block_17:
        var_r6 = 3;
    }
    if (arg0 != 0) {
        if ((arg1 != 0) && ((arg2 != 0) || (arg3 != 0)) && ((var_r2 != 0) || ((arg1 + 1) != 0))) {
            var_r6 = 4;
        }
        if (arg0 == 0) {
            goto block_26;
        }
        goto block_27;
    }
block_26:
    if (arg1 != 0) {
block_27:
        if (arg2 == 0) {
            goto block_28;
        }
        goto block_32;
    }
block_28:
    if (((arg3 != 0) && (var_r2 != 0)) || ((arg1 + 1) != 0) || ((arg2 + 1) != 0)) {
block_32:
        var_r6 = 5;
    }
    if ((((arg0 != 0) && (arg1 != 0)) || ((arg2 != 0) && (arg3 != 0))) && ((var_r2 != 0) || ((arg1 + 1) != 0))) {
        var_r6 = 6;
    }
    if (arg0 != 0) {
        if (arg1 != 0) {
            if (arg2 != var_r2) {

            } else {
                goto block_45;
            }
        } else if (arg3 == var_r2) {
block_45:
            if ((arg1 + 1) != 0) {
                var_r6 = 7;
            }
        }
        if (arg0 == 0) {
            goto block_48;
        }
        goto block_53;
    }
block_48:
    if (arg1 != 0) {
        if (arg2 != var_r2) {
            goto block_52;
        }
        goto block_53;
    }
    if (arg3 != var_r2) {
block_52:
        if ((arg1 + 1) != 0) {
            goto block_53;
        }
    } else {
block_53:
        var_r6 = 8;
    }
    return var_r6;
}
