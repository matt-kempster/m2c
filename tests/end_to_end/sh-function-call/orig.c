int callee(int first, int second);

int test(int value) {
    return callee(value, value + 1) + 3;
}
