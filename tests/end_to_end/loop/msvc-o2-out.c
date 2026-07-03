void test(s32 arg0, s32 arg1) {
    u32 temp_ecx;

    if (arg1 > 0) {
        temp_ecx = (u32) arg1 >> 2;
        M2C_MEMSET32(arg0, 0, temp_ecx);
        M2C_MEMSET(arg0 + (temp_ecx * 4), 0, arg1 & 3);
    }
}
