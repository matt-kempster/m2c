f32 test(f32 arg0)
{
    s32 spC;
    s32 sp4;
    f32 temp_f0;
    f32 temp_f4;

    spC = sp4;
    spC = (s32) (0x5F3759DF - (sp4 >> 1));
    temp_f4 = (bitwise f32) spC;
    sp4 = temp_f4;
    temp_f0 = (1.5f - (((arg0 * 0.5f) * temp_f4) * temp_f4)) * temp_f4;
    sp4 = temp_f0;
    return temp_f0;
}
