typedef struct {
    int a;
    int b;
    int c;
} Large;

Large test(int value) {
    Large ret = {value, value + 1, value + 2};
    return ret;
}
