extern u32 global;

void test(u32 arg0, s32 arg1, s32 arg2) {
    s32 temp_r0;

    temp_r0 = arg1 ^ arg0;
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = (u32) ((temp_r0 >> 1) - (temp_r0 & arg1)) >> 0x1FU;
    global = (arg1 >> 0x1F) + (arg0 >> 0x1FU) + M2C_CARRY;
    global = arg0 == 0;
    global = arg1 != 0;
}
