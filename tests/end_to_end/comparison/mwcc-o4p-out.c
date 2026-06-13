extern s32 global;

void test(s32 arg0, s32 arg1, s32 arg2) {
    global = arg0 == arg1;
    global = arg0 != arg2;
    global = ((s32) (u32) ~(arg1 ^ arg0) / 2147483648) & 1;
    global = (arg1 >> 0x1F) + ((u32) arg0 >> 0x1FU) + M2C_CARRY;
    global = -arg0 == 0;
    global = arg1 != 0;
}
