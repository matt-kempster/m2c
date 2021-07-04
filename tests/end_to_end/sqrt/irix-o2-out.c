f32 test(f32 arg0) {
    f32 spC;
    f32 sp4;
    f32 temp_f0;

    sp4 = arg0;
    spC = sp4;
    spC = (bitwise f32) (0x5F3759DF - ((bitwise s32) sp4 >> 1));
    sp4 = spC;
    temp_f0 = (1.5f - (arg0 * 0.5f * spC * spC)) * spC;
    sp4 = temp_f0;
    return temp_f0;
}
