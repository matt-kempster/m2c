s32 __eqsf2(s32, s32);                              /* extern */

s32 test(s32 arg0) {
    s32 var_r1;

    var_r1 = 0x41700000;
    if (__eqsf2(arg0, 0) == 0) {
        var_r1 = arg0;
    }
    return var_r1;
}
