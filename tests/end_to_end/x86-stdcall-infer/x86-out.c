? GlobalFree(s32);                                  /* extern */

void test(s32 arg0) {
    s32 var_edi;

    var_edi = 0;
    do {
        GlobalFree(arg0);
        var_edi += 1;
    } while (var_edi < 0xA);
}
