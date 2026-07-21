? bar();                                            /* extern */
? foo();                                            /* extern */

s32 test(s32 arg0) {
    s32 sp0;

    sp0 = arg0 + 1;
    foo();
    sp0 = 0x4D2;
    bar();
    return sp0;
}
