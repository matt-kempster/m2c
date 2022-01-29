extern s32 glob;

? test(void) {
    s32 temp_cr0_lt;
    s32 temp_r3;

    temp_r3 = glob;
    temp_cr0_lt = temp_r3 < 1;
    glob = temp_r3 - 1;
    if (temp_cr0_lt != 0) {
        return 4;
    }
    return 6;
}
