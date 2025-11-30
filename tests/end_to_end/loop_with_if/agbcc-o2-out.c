void test(s32 arg0) {
    s32 var_r0;

    var_r0 = 0;
loop_1:
    if (var_r0 < arg0) {
        if (var_r0 == 5) {
            var_r0 = 0xA;
        } else {
            var_r0 += 4;
        }
        goto loop_1;
    }
}
