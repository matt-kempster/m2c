? foo(s32);                                         /* static */

s32 test(s32 arg0) {
    s32 temp_r4;
    s32 var_r4;

    var_r4 = arg0;
loop_2:
    foo(var_r4);
    temp_r4 = var_r4 * 2;
    if (temp_r4 <= 3) {
        foo(temp_r4);
        var_r4 = temp_r4 + 1;
        goto loop_2;
    }
    return temp_r4;
}
