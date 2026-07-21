int callee(int a, int b, int c, int d, int e);
int callee6(int a, int b, int c, int d, int e, int f);
int callee7(int a, int b, int c, int d, int e, int f, int g);
int callee8(int a, int b, int c, int d, int e, int f, int g, int h);

int test(int value) {
    return callee(value, value + 1, value + 2, value + 3, value + 4) + 5;
}

int test6(int value) {
    return callee6(value, value + 1, value + 2, value + 3, value + 4, value + 5);
}

int test7(int value) {
    return callee7(
        value, value + 1, value + 2, value + 3, value + 4, value + 5, value + 6
    );
}

int test8(int value) {
    return callee8(
        value,
        value + 1,
        value + 2,
        value + 3,
        value + 4,
        value + 5,
        value + 6,
        value + 7
    );
}
