extern s32 glob;

s32 test(void) {
    s32 temp_r2;

    temp_r2 = glob;
    glob -= 1;
    if (temp_r2 > 0) {
        return 6;
    }
    return 4;
}
