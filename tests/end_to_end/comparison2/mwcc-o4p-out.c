extern s32 global;

void test(u32 arg0, s32 arg1, s32 arg2) {
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = arg0 <= arg1;
    global = arg1 < arg0;
    global = (u32) arg1 <= (s32) arg0;
    global = arg0 == 0;
    global = arg1 != 0;
}
