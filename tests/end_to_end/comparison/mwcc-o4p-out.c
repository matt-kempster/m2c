extern u32 global;

void test(s32 arg0, s32 arg1, s32 arg2) {
    s32 temp_r0;

    global = arg0 == arg1;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = arg0 <= arg1;
    global = arg0 == 0;
    global = arg1 != 0;
    temp_r0 = CLZ(arg1);
    global = (u32) (-arg1 & ~arg1) >> 0x1FU;
    global = ((1 << (temp_r0 & 0x1F)) & 1) | ((1 >> (0x20 - (temp_r0 & 0x1F))) & 1);
}
