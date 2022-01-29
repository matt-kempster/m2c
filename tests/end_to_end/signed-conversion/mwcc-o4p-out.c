f32 test(s8 arg0, f32 arg8) {
    s8 temp_r5;
    s8 temp_r6;

    temp_r5 = arg0 * 2;
    *NULL = (s32) (s8) arg0;
    temp_r6 = arg0 * 3;
    *NULL = (s32) (s8) temp_r5;
    *NULL = (s32) (s8) temp_r6;
    *NULL = (s32) (s16) arg0;
    *NULL = (s32) (s16) temp_r5;
    *NULL = (s32) (s16) temp_r6;
    return arg8;
}
