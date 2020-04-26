int* foo(int* x, int y) { return x; }

void test(int x, int y, int z) {
    int* ptr = (void*) 0;
    for (;;) {
        ptr = foo(ptr, y);
    }
}
