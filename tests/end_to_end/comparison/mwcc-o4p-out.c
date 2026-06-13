extern s32 global;

void test(s32 arg0, s32 arg1, s32 arg2) {
    global = arg0 == arg1;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = arg0 <= arg1;
    global = -arg0 == 0;
    global = arg1 != 0;
}
