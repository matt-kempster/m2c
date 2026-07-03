? foo(s32);                                         /* static */

s32 test(s32 arg0) {
    s32 temp_esi;
    s32 var_esi;

    var_esi = arg0;
loop_1:
    foo(var_esi);
    temp_esi = var_esi * 2;
    if (temp_esi < 4) {
        foo(temp_esi);
        var_esi = temp_esi + 1;
        goto loop_1;
    }
    return temp_esi;
}
