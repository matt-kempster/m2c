? foo(s32);                                         /* static */

void test(s32 arg0) {
    if (arg0 == 7) {
        foo(1);
        foo(3);
        return;
    }
    foo(4);
}
