s32 test(s32 arg0, s32 arg1, s32 arg2) {
    s32 temp_eax;
    s32 temp_eax_2;

    temp_eax_2 = (arg0 + arg1) - (arg2 * 3);
    temp_eax = (temp_eax_2 / arg1) + -(temp_eax_2 % arg1);
    return (temp_eax * arg1) + MULTU_HI(temp_eax, arg1);
}
