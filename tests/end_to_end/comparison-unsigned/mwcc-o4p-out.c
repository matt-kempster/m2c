extern u32 global;

void test(s32 arg0, s32 arg1) {
    global = (u32) (arg1 << CLZ(arg1 ^ arg0)) >> 0x1FU;
    global = (u32) ((arg1 | ~arg0) - ((u32) (arg1 - arg0) >> 1U)) >> 0x1FU;
}
