extern int (float) bar;
extern int (*)(float) glob2;

void test(void) {
    glob = foo;
    glob = &bar;
    glob2 = (int (*)(float)) foo;
    glob2 = &bar;
}
