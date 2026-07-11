void test(s32 x) {
    s32 var_eax;

    var_eax = x;
loop_1:
    if (var_eax <= 2) {
        goto loop_1;
    }
    var_eax = foo(var_eax);
    goto loop_1;
}
