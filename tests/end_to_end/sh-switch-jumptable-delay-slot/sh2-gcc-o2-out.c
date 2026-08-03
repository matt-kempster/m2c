s32 _sink(s32, ?, u32, s32);                        /* extern */

s32 test(s32 arg0, s32 arg1) {
    s32 var_r3;
    s32 var_r6;
    s32 var_r7;
    s32 var_r8;

    var_r8 = arg1;
    var_r6 = 0;
    var_r3 = 1;
    var_r7 = 4;
    do {
        if (*(arg0 + var_r7) > *(arg0 + (var_r6 * 4))) {
            var_r6 = var_r3;
        }
        var_r3 += 1;
        var_r7 += 4;
    } while (var_r3 <= 3);
    switch (var_r6) {
    case 0:
        var_r8 *= 2;
        break;
    case 1:
        var_r8 += 3;
        break;
    case 2:
        var_r8 += sink(var_r8, 3, (u32) var_r6, var_r7);
        break;
    case 3:
        var_r8 -= sink(var_r8, 3, (u32) var_r6, var_r7);
        break;
    }
    return var_r8;
}
