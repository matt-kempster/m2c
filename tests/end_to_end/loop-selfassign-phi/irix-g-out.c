s32 func_004000D8(s32); // static
void test(s32 arg0); // static

void test(s32 arg0) {
loop_1:
    if (arg0 < 3) {
        goto block_3;
    }
    arg0 = func_004000D8(arg0);
block_3:
    goto loop_1;
}
