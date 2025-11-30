extern s32 x;

void test(void) {
    if (x != 2) {
loop_2:
        x = 1;
        goto loop_2;
    }
}
