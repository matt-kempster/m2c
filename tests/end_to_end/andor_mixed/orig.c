int test(int a, int b, int c, int d) {
    int var1;
    int var2;
    int ret;

    var1 = a + b;
    var2 = b + c;

    ret = 0;
    if (var1 || (var2 && (a * b)) || (d && a)) {
        ret = a + d;
    }

    return ret;
}
se