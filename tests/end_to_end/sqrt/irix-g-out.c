f32 test(f32 arg0)
{
    s32 spC;
    f32 sp8;
    f32 sp4;
    f32 sp0;

    sp0 = 1.5f;
    sp8 = arg0 * 0.5f;
    sp4 = arg0;
    spC = (bitwise s32) sp4;
    spC = 0x5F3759DF - (spC >> 1);
    sp4 = (bitwise f32) spC;
    sp4 = (sp0 - ((sp8 * sp4) * sp4)) * sp4;
    return sp4;
}
