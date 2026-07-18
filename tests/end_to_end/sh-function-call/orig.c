int callee(int first, int second);
extern int global;

int test(int value) {
    return callee(value, value + 1) + global;
}
