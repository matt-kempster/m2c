? foo(s8 *);

void test(struct A *a) {
    foo(&a->b);
}
