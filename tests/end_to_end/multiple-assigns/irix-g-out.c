s32 test(s32 arg0)
{
    s32 sp4;
    s32 temp_a0;
    s32 phi_a0;

    phi_a0 = arg0;
    if (arg0 == 5)
    {
loop_1:
        D_410150 = (s32) phi_a0;
        temp_a0 = phi_a0 + 1;
        D_410150 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410150 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410150 = temp_a0;
        sp4 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410150 = temp_a0;
        D_410150 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410150 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410150 = sp4;
        phi_a0 = temp_a0;
        if (temp_a0 == 5)
        {
            goto loop_1;
        }
    }
    return sp4;
}
