void test_s16(s16 arg0, s16 arg1, s16 arg2);        /* static */
void test_s32(u32 arg0, s32 arg1, s32 arg2);        /* static */
void test_u16(s16 arg0, s16 arg1, s16 arg2);        /* static */
void test_u32(s32 arg0, s32 arg1, s32 arg2);        /* static */
extern u32 global;

void test(void) {
    test_s32(1U, 2, 3);
    test_u32(1, 2, 3);
    test_s16(1, 2, 3);
    test_u16(1, 2, 3);
}

void test_s32(u32 arg0, s32 arg1, s32 arg2) {
    s32 temp_r0;

    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = arg0 <= arg1;
    global = arg1 < arg0;
    global = (u32) arg1 <= (s32) arg0;
    global = arg0 == 0;
    global = arg1 != 0;
    global = arg0 >> 0x1FU;
    temp_r0 = CLZ(arg0);
    global = ((1 << (temp_r0 & 0x1F)) & 1) | ((1 >> (0x20 - (temp_r0 & 0x1F))) & 1);
    global = (u32) (-(s32) arg0 & ~arg0) >> 0x1FU;
    global = (arg0 >> 0x1FU) ^ 1;
}

void test_u32(s32 arg0, s32 arg1, s32 arg2) {
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = (u32) ((arg1 | ~arg0) - ((u32) (arg1 - arg0) >> 1U)) >> 0x1FU;
    global = arg1 < arg0;
    global = (u32) ((arg0 | ~arg1) - ((u32) (arg0 - arg1) >> 1U)) >> 0x1FU;
    global = arg0 == 0;
    global = arg1 != 0;
    global = 0;
    global = arg0 == 0;
    global = arg0 != 0;
    global = 1;
}

void test_s16(s16 arg0, s16 arg1, s16 arg2) {
    s32 temp_r0;

    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = (u32) arg0 <= arg1;
    global = arg1 < arg0;
    global = (u32) arg1 <= arg0;
    global = arg0 == 0;
    global = arg1 != 0;
    global = (u32) arg0 >> 0x1FU;
    temp_r0 = CLZ(arg0);
    global = ((1 << (temp_r0 & 0x1F)) & 1) | ((1 >> (0x20 - (temp_r0 & 0x1F))) & 1);
    global = (u32) (-arg0 & ~arg0) >> 0x1FU;
    global = ((u32) arg0 >> 0x1FU) ^ 1;
}

void test_u16(s16 arg0, s16 arg1, s16 arg2) {
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = (u32) (arg0 - arg1) >> 0x1FU;
    global = (u32) ((arg1 | ~arg0) - ((u32) (arg1 - arg0) >> 1U)) >> 0x1FU;
    global = (u32) (arg1 - arg0) >> 0x1FU;
    global = (u32) ((arg0 | ~arg1) - ((u32) (arg0 - arg1) >> 1U)) >> 0x1FU;
    global = arg0 == 0;
    global = arg1 != 0;
    global = 0;
    global = arg0 == 0;
    global = arg0 != 0;
    global = 1;
}
