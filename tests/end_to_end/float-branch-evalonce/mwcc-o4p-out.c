s32 test(s32 arg0) {
    s32 temp_cr0_lt;
    s32 temp_cr0_lt_2;

    temp_cr0_lt = *NULL < *NULL;
    *NULL = (f32) *NULL;
    if (temp_cr0_lt != 0) {
        *NULL = (f32) *NULL;
    }
    temp_cr0_lt_2 = *NULL < *NULL;
    *NULL = (f32) *NULL;
    if (temp_cr0_lt_2 == 0) {
        *NULL = (f32) *NULL;
        return arg0;
    }
    return arg0;
}
