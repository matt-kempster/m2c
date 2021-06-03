void test(void) {
    glob = (int (*)(float)) foo;
    glob = &bar;
    glob2 = (int (*)(float)) foo;
    glob2 = (int (*)(float)) &bar;
}
