s32 test(s32 arg0)
{
    s32 sp4;
    s32 temp_a0;
    s32 temp_v1;
    s32 phi_a0;

    if (arg0 == 5)
    {
        phi_a0 = arg0;
loop_2:
        D_410120 = (s32) phi_a0;
        temp_a0 = phi_a0 + 1;
        D_410120 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410120 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410120 = temp_a0;
        temp_v1 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410120 = temp_a0;
        D_410120 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410120 = temp_a0;
        temp_a0 = temp_a0 + 1;
        D_410120 = temp_v1;
        phi_a0 = temp_a0;
        if (temp_a0 == 5)
        {
            goto loop_2;
        }
        sp4 = temp_v1;
    }
    return sp4;
}
