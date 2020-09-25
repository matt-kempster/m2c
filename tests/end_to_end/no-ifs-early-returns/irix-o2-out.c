void test(void *arg0, void *arg1) {
    s32 temp_v1;

    temp_v1 = *arg0;
    if (temp_v1 == 8) {
        goto block_3;
    }
    if (temp_v1 == 0xF) {
        goto block_4;
    }
block_4:
    *arg1 = (s32) (*arg1 - temp_v1);
}
