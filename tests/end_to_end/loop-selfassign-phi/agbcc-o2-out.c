s32 foo();                                          /* static */

void test(s32 arg0) {
    s32 var_r0;

    var_r0 = arg0;
loop_1:
    if (var_r0 <= 2) {
        goto loop_1;
    }
    var_r0 = foo();
    goto loop_1;
}
