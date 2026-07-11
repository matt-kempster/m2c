void test(void *a, void *b) {
    a = &a;
    b = &b;
    foo(&a, &b);
    a = b;
    foo((void **) b, (void **) b);
}
