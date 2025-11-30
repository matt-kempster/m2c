u16 foo(s32);                                       /* static */
extern s32 glob;

u16 test(void) {
    s32 var_r0_2;
    u16 var_r0;

    var_r0 = foo(1);
    if (var_r0 == 0) {
        if (glob == 0x7B) {
            var_r0_2 = 3;
        } else {
            var_r0_2 = 2;
        }
        var_r0 = foo(var_r0_2);
    }
    return var_r0;
}
