s32 foo(s32);                                       /* static */

void test(s32 arg0) {
    s32 var_eax;

    var_eax = arg0;
loop_1:
    if (var_eax <= 2) {
        goto loop_1;
    }
    var_eax = foo(var_eax);
    goto loop_1;
}
