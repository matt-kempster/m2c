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
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = arg0 <= arg1;
    global = arg1 < arg0;
    global = (u32) arg1 <= (s32) arg0;
    global = arg0 == 0;
    global = arg1 != 0;
    global = arg0 >> 0x1FU;
    global = M2C_ERROR(/* unknown instruction: rlwnm $r0, $r6, $r0, 0x1f, 0x1f */);
    global = (u32) (-(s32) arg0 & ~arg0) >> 0x1FU;
    global = (arg0 >> 0x1FU) ^ 1;
}

void test_u32(s32 arg0, s32 arg1, s32 arg2) {
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = (u32) (M2C_ERROR(/* unknown instruction: orc $r6, $r4, $r3 */) - ((u32) (arg1 - arg0) >> 1U)) >> 0x1FU;
    global = arg1 < arg0;
    global = (u32) (M2C_ERROR(/* unknown instruction: orc $r6, $r3, $r4 */) - ((u32) (arg0 - arg1) >> 1U)) >> 0x1FU;
    global = arg0 == 0;
    global = arg1 != 0;
    global = 0;
    global = arg0 == 0;
    global = arg0 != 0;
    global = 1;
}

void test_s16(s16 arg0, s16 arg1, s16 arg2) {
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = (u32) arg0 <= arg1;
    global = arg1 < arg0;
    global = (u32) arg1 <= arg0;
    global = arg0 == 0;
    global = arg1 != 0;
    global = (u32) arg0 >> 0x1FU;
    global = M2C_ERROR(/* unknown instruction: rlwnm $r0, $r6, $r0, 0x1f, 0x1f */);
    global = (u32) (-arg0 & ~arg0) >> 0x1FU;
    global = ((u32) arg0 >> 0x1FU) ^ 1;
}

void test_u16(s16 arg0, s16 arg1, s16 arg2) {
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = (u32) (arg0 - arg1) >> 0x1FU;
    global = (u32) (M2C_ERROR(/* unknown instruction: orc $r6, $r6, $r7 */) - ((u32) (arg1 - arg0) >> 1U)) >> 0x1FU;
    global = (u32) (arg1 - arg0) >> 0x1FU;
    global = (u32) (M2C_ERROR(/* unknown instruction: orc $r6, $r7, $r6 */) - ((u32) (arg0 - arg1) >> 1U)) >> 0x1FU;
    global = arg0 == 0;
    global = arg1 != 0;
    global = 0;
    global = arg0 == 0;
    global = arg0 != 0;
    global = 1;
}

void test_s8(s8 arg0, s8 arg1, s8 arg2) {
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = arg0 < arg1;
    global = (u32) arg0 <= arg1;
    global = arg1 < arg0;
    global = (u32) arg1 <= arg0;
    global = arg0 == 0;
    global = arg1 != 0;
    global = (u32) arg0 >> 0x1FU;
    global = M2C_ERROR(/* unknown instruction: rlwnm $r0, $r6, $r0, 0x1f, 0x1f */);
    global = (u32) (-arg0 & ~arg0) >> 0x1FU;
    global = ((u32) arg0 >> 0x1FU) ^ 1;
}

void test_u8(s8 arg0, s8 arg1, s8 arg2) {
    global = arg1 == arg0;
    global = arg0 != arg2;
    global = (u32) (arg0 - arg1) >> 0x1FU;
    global = (u32) (M2C_ERROR(/* unknown instruction: orc $r6, $r6, $r7 */) - ((u32) (arg1 - arg0) >> 1U)) >> 0x1FU;
    global = (u32) (arg1 - arg0) >> 0x1FU;
    global = (u32) (M2C_ERROR(/* unknown instruction: orc $r6, $r7, $r6 */) - ((u32) (arg0 - arg1) >> 1U)) >> 0x1FU;
    global = arg0 == 0;
    global = arg1 != 0;
    global = 0;
    global = arg0 == 0;
    global = arg0 != 0;
    global = 1;
}
