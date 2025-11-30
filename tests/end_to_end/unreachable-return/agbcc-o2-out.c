extern s32 x;

void test(void) {
loop_1:
    x = 1;
    goto loop_1;
}
