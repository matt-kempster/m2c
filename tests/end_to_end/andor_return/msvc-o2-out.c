s32 test(s32 a, s32 b, s32 c, s32 d) {
    if (((a == 0) && (b == 0)) || ((c == 0) && (d == 0))) {
        return c + d;
    }
    return a + b;
}
