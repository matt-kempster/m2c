extern s32 global;

void test(u32 arg0, s32 arg1, s32 arg2) {
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = (arg1 >> 0x1F) + (arg0 >> 0x1FU) + M2C_CARRY;
    global = arg1 < arg0;
    global = ((s32) arg0 >> 0x1F) + ((u32) arg1 >> 0x1FU) + M2C_CARRY;
    global = arg0 == 0;
    global = arg1 != 0;
}
