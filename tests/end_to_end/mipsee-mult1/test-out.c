s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    return (arg0 * arg1) + (arg2 * arg3) + MULTU_HI(arg0, arg1) + MULTU_HI(arg2, arg3);
}

s32 test2(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    return (arg0 * arg1) + (arg2 * arg3) + (arg2 * arg3);
}

s32 test3(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 temp_t2;

    temp_t2 = (s32) (arg1 + (arg2 * arg3) + (arg2 * arg3)) / arg0;
    return ((u32) temp_t2 / (u32) arg0) + ((u32) temp_t2 % (u32) arg0) + (arg0 * arg0) + MULT_HI(arg0, arg0);
}
