extern s32 global;

void test(u32 arg0, u32 arg1) {
    global = arg0 < arg1;
    global = arg0 <= arg1;
}
