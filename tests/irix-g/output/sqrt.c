f32 test(f32 arg0)
{
    s32 spC;
    f32 sp8;
    f32 sp4;
    f32 sp0;

    sp0 = 1.5f;
    sp8 = (f32) (arg0 * 0.5f);
    spC = (s32) sp4;
    spC = (s32) ((0x5f370000 | 0x59df) - (spC >> 1));
    sp4 = (f32) spC;
    sp4 = (f32) ((sp0 - ((sp8 * sp4) * sp4)) * sp4);
    return sp4;
}
