void test(s8 *foo, s32 length) {
    u32 temp_ecx;

    if (length > 0) {
        temp_ecx = (u32) length >> 2;
        M2C_MEMSET32(foo, 0, temp_ecx);
        M2C_MEMSET(foo + (temp_ecx * 4), 0, length & 3);
    }
}
