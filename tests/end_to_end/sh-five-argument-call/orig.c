int callee(int a, int b, int c, int d, int e);

int test(int value) {
    return callee(value, value + 1, value + 2, value + 3, value + 4) + 5;
}
