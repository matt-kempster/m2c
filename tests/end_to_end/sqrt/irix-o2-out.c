f32 test(f32 arg0)
{
    s32 spC;
    s32 sp4;
    f32 temp_f0;

    spC = sp4;
    spC = (s32) (0x5f3759df - (sp4 >> 1));
    sp4 = (bitwise f32) spC;
    temp_f0 = (1.5f - (((arg0 * 0.5f) * (bitwise f32) spC) * (bitwise f32) spC)) * (bitwise f32) spC;
    sp4 = temp_f0;
    return temp_f0;
}
