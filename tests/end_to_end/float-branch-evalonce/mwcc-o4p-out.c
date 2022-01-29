extern f32 x;

s32 test(s32 arg0) {
    s32 temp_cr0_lt;
    s32 temp_cr0_lt_2;

    temp_cr0_lt = x < 0.0f;
    x = 5.0f;
    if (temp_cr0_lt != 0) {
        x = 6.0f;
    }
    temp_cr0_lt_2 = x < 0.0f;
    x = 3.0f;
    if (temp_cr0_lt_2 == 0) {
        x = 7.0f;
        return arg0;
    }
    return arg0;
}
