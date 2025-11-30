void test(s32 *arg0, s32 *arg1) {
    s32 temp_r0;
    s32 var_r0;

    temp_r0 = *arg0;
    if (temp_r0 == 8) {
        goto block_3;
    }
    if (temp_r0 == 0xF) {
        goto block_4;
    }
    goto block_6;
block_3:
    var_r0 = *arg1 + 8;
    goto block_5;
block_4:
    var_r0 = *arg1 - 0xF;
block_5:
    *arg1 = var_r0;
block_6:
    return;
}
