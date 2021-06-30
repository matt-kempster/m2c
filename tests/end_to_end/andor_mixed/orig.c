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

    if (a && (b || c) && (d || (a + 1))) {
        ret = b + d;
    }

    if ((a && d) || ((b || c) && (d + c))) {
        ret = c + d;
    }

    return ret;
}
