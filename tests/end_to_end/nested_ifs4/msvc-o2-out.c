void test(s32 x) {
    if (x == 7) {
        foo(1);
        foo(3);
        return;
    }
    foo(4);
    if (x == 9) {
        foo(5);
    }
    foo(6);
}
