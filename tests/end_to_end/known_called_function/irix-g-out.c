void test(s32 x, short *y, s32 z, char *r, short *s, int *t, long *u) {
    int *sp1C;

    sp1C = NULL;
loop_1:
    sp1C = foo(sp1C, y, t);
    goto loop_1;
}
