s8 foo(s8);                                         /* extern */

s8 test(s32 arg0) {
    s32 sp10;

    sp10 = arg0;
    return (s8) (foo((s8) sp10) + 1);
}
