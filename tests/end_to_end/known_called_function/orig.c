int* foo(int* x, short* y) { return x; }

void test(int x, short* y, int z) {
    int* ptr = (void*) 0;
    for (;;) {
        ptr = foo(ptr, y);
    }
}
