void test(void) {
    glob = foo;
    glob = &bar;
    glob2 = (int (*)(float) *) foo;
    glob2 = &bar;
}
