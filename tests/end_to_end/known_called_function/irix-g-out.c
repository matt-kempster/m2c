void test(s32 x, short *y, s32 z, char *r, int *arg5) {
    int *sp1C;

    sp1C = NULL;
loop_1:
    sp1C = foo(sp1C, y, arg5);
    goto loop_1;
}
