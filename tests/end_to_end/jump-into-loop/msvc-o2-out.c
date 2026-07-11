s32 test(s32 x) {
    s32 temp_esi;
    s32 var_esi;

    var_esi = x;
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
