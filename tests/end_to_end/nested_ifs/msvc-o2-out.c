void test(s32 x) {
    if (x == 7) {
        foo(1);
        return;
    }
    foo(2);
    if (x == 8) {
        foo(3);
    }
    foo(4);
}
