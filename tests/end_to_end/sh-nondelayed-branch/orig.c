void test(int a, int b, int *out) {
    if (a)
        *out = a;
    if (b)
        *out = b;
}

void test_bf(int a, int b, int *out) {
    if (!a)
        *out = a;
    if (!b)
        *out = b;
}
