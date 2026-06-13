extern s32 global;

void test(s32 arg0, s32 arg1) {
    s32 temp_r0;

    temp_r0 = arg0 - arg1;
    global = -((temp_r0 - temp_r0) - !M2C_CARRY);
    global = --1 - !M2C_CARRY;
}
