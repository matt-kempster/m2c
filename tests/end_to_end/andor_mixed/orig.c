int test(int a, int b, int c, int d) {
    int var1;
    int var2;
    int ret;

    var1 = a + b;
    var2 = b + c;

    ret = 0;
    if (var1 || (var2 && (a * b)) || (d && a)) {
        ret = 1;
    }

    if (a && (b || c) && (d || (a + 1))) {
        ret = 2;
    }

    if ((a && d) || ((b || c) && (d + c))) {
        ret = 3;
    }

    if (a && (b && (c || d)) && (a + d || b + c)) {
        ret = 4;
    }

    if (((a || b) && c) || (d && (a + 1)) || (a + 2) || (a + 3)) {
        ret = 5;
    }

    return ret;
}
