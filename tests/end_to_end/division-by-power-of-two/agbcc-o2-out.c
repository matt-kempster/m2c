void test(u32 *arg0) {
    u32 temp_r1;

    temp_r1 = *arg0;
    *arg0 = (u32) ((s32) (temp_r1 + (temp_r1 >> 0x1F)) >> 1);
}
