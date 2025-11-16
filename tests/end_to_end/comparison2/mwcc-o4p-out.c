void test_s16(s16 arg0, s16 arg1, s16 arg2);        /* static */
void test_s32(u32 arg0, s32 arg1, s32 arg2);        /* static */
void test_s8(s8 arg0, s8 arg1, s8 arg2);            /* static */
void test_u16(s16 arg0, s16 arg1, s16 arg2);        /* static */
void test_u32(s32 arg0, s32 arg1, s32 arg2);        /* static */
void test_u8(s8 arg0, s8 arg1, s8 arg2);            /* static */
extern u32 global;

void test(void) {
    test_s32(1U, 2, 3);
    test_u32(1, 2, 3);
    test_s16(1, 2, 3);
    test_u16(1, 2, 3);
    test_s8(1, 2, 3);
    test_u8(1, 2, 3);
}

void test_s32(u32 arg0, s32 arg1, s32 arg2) {
    u32 temp_r7;
    u32 temp_r9;

    temp_r9 = arg0 >> 0x1FU;
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = arg0 <= arg1;
    temp_r7 = CLZ(arg0);
    global = arg1 < arg0;
    global = (u32) arg1 <= (s32) arg0;
    global = arg0 == 0;
    global = arg1 != 0;
    global = temp_r9;
    global = ((1 << (temp_r7 & 0x1F)) & 1) | ((1 >> (0x20 - (temp_r7 & 0x1F))) & 1);
    global = (u32) (-(s32) arg0 & ~arg0) >> 0x1FU;
    global = temp_r9 ^ 1;
}

void test_u32(s32 arg0, s32 arg1, s32 arg2) {
    u32 temp_r6;

    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = (u32) ((arg1 | ~arg0) - ((u32) (arg1 - arg0) >> 1U)) >> 0x1FU;
    global = (u32) (arg0 << CLZ(arg1 ^ arg0)) >> 0x1FU;
    temp_r6 = arg0 == 0;
    global = (u32) ((arg0 | ~arg1) - ((u32) (arg0 - arg1) >> 1U)) >> 0x1FU;
    global = temp_r6;
    global = arg1 != 0;
    global = 0;
    global = temp_r6;
    global = arg0 != 0;
    global = 1;
}

void test_s16(s16 arg0, s16 arg1, s16 arg2) {
    u32 temp_r7;
    u32 temp_r8;

    global = arg1 == arg0;
    temp_r8 = (u32) arg0 >> 0x1FU;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = (u32) arg0 <= arg1;
    temp_r7 = CLZ(arg0);
    global = arg1 < arg0;
    global = (u32) arg1 <= arg0;
    global = arg0 == 0;
    global = arg1 != 0;
    global = temp_r8;
    global = ((1 << (temp_r7 & 0x1F)) & 1) | ((1 >> (0x20 - (temp_r7 & 0x1F))) & 1);
    global = (u32) (-arg0 & ~arg0) >> 0x1FU;
    global = temp_r8 ^ 1;
}

void test_u16(s16 arg0, s16 arg1, s16 arg2) {
    u32 temp_r6;
    u32 temp_r7;
    u32 temp_r8;

    temp_r8 = arg1 - arg0;
    temp_r7 = arg0 - arg1;
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = temp_r7 >> 0x1FU;
    global = (u32) ((arg1 | ~arg0) - (temp_r8 >> 1U)) >> 0x1FU;
    global = temp_r8 >> 0x1FU;
    global = (u32) ((arg0 | ~arg1) - (temp_r7 >> 1U)) >> 0x1FU;
    temp_r6 = arg0 == 0;
    global = temp_r6;
    global = arg1 != 0;
    global = 0;
    global = temp_r6;
    global = arg0 != 0;
    global = 1;
}

void test_s8(s8 arg0, s8 arg1, s8 arg2) {
    u32 temp_r7;
    u32 temp_r8;

    global = arg1 == arg0;
    temp_r8 = (u32) arg0 >> 0x1FU;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = (u32) arg0 <= arg1;
    temp_r7 = CLZ(arg0);
    global = arg1 < arg0;
    global = (u32) arg1 <= arg0;
    global = arg0 == 0;
    global = arg1 != 0;
    global = temp_r8;
    global = ((1 << (temp_r7 & 0x1F)) & 1) | ((1 >> (0x20 - (temp_r7 & 0x1F))) & 1);
    global = (u32) (-arg0 & ~arg0) >> 0x1FU;
    global = temp_r8 ^ 1;
}

void test_u8(s8 arg0, s8 arg1, s8 arg2) {
    u32 temp_r6;
    u32 temp_r7;
    u32 temp_r8;

    temp_r8 = arg1 - arg0;
    temp_r7 = arg0 - arg1;
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = temp_r7 >> 0x1FU;
    global = (u32) ((arg1 | ~arg0) - (temp_r8 >> 1U)) >> 0x1FU;
    global = temp_r8 >> 0x1FU;
    global = (u32) ((arg0 | ~arg1) - (temp_r7 >> 1U)) >> 0x1FU;
    temp_r6 = arg0 == 0;
    global = temp_r6;
    global = arg1 != 0;
    global = 0;
    global = temp_r6;
    global = arg0 != 0;
    global = 1;
}
