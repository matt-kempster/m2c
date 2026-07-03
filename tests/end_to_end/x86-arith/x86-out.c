s32 test(s32 arg0, s32 arg1, s32 arg2) {
    s32 temp_ecx;

    temp_ecx = (arg0 - (arg2 * 0x8C)) + arg1;
    return (temp_ecx / arg1) + MULTU_HI(-(temp_ecx % arg1), arg1);
}
