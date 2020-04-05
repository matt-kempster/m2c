f32 test(f32 arg0)
{
    s32 spC;
    f32 sp4;
    f32 temp_f0;
    f32 temp_f4;
    s32 temp_t6;

    sp4 = arg0;
    temp_t6 = (bitwise s32) sp4;
    spC = temp_t6;
    spC = 0x5F3759DF - (temp_t6 >> 1);
    temp_f4 = (bitwise f32) spC;
    sp4 = temp_f4;
    temp_f0 = (1.5f - (((arg0 * 0.5f) * temp_f4) * temp_f4)) * temp_f4;
    sp4 = temp_f0;
    return temp_f0;
}
