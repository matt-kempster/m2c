s16 foo(s16);                                       /* extern */

s16 test(s32 arg0) {
    s32 sp10;

    sp10 = arg0;
    return (s16) (foo((s16) sp10) + 1);
}
