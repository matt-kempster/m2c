void test(u32 *arg0) {
    u32 temp_r2;

    temp_r2 = *arg0;
    *arg0 = temp_r2 - (((s32) (temp_r2 + (temp_r2 >> 0x1F)) >> 1) * 2);
}
