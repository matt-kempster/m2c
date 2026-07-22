s32 test(s32 value) {
    if (value != 0) {
        return 2;
    }
    return 1;
}

s32 test_if_nonzero(s32 value) {
    if (value == 0) {
        return 4;
    }
    return 3;
}

s32 test_if_greater(s32 lhs, s32 rhs) {
    s32 var_r0;

    var_r0 = rhs;
    if (lhs > var_r0) {
        var_r0 = lhs;
    }
    return var_r0;
}

s32 test_loop(s32 value) {
    s32 var_r0;
    s32 var_r0_2;
    s32 var_r4;

    var_r4 = value;
    var_r0 = 0;
    if (var_r4 != 0) {
        var_r0_2 = var_r4;
        do {
            var_r4 -= 1;
            var_r0_2 += var_r4;
        } while (var_r4 != 0);
        var_r0 = var_r0_2 - var_r4;
    }
    return var_r0;
}

s32 test_call(s32 value) {
    return callee(value) + 1;
}
