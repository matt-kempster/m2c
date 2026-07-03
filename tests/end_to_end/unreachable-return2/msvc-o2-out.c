extern s32 _x;

void test(void) {
    if (_x != 2) {
loop_1:
        goto loop_1;
    }
}
