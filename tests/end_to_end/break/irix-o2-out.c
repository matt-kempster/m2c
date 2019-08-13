s32 test(s32 arg0)
{
    s32 temp_v0;
    s32 phi_v0;
    s32 phi_return;

    phi_return = 0;
    if (arg0 > 0)
    {
        phi_v0 = 0;
loop_2:
        temp_v0 = phi_v0 + 1;
        D_410150.unk0 = 1;
        if (2 == D_410150.unk4)
        {
            D_410150.unk8 = 3;
            phi_return = temp_v0;
        }
        else
        {
            if (2 == D_410150.unk8)
            {
                D_410150.unkC = 3;
block_11:
                phi_v0 = temp_v0;
                phi_return = temp_v0;
                if (temp_v0 != arg0)
                {
                    goto loop_2;
                }
            }
            else
            {
                if (2 == D_410150.unk10)
                {
                    D_410150.unk14 = 3;
                    phi_return = temp_v0;
                }
                else
                {
                    if (2 == D_410150.unk14)
                    {
                        D_410150.unk18 = 3;
                    }
                    else
                    {
                        D_410150.unkC = 4;
                    }
                    goto block_11;
                }
            }
        }
    }
    D_410150.unk10 = 5;
    return phi_return;
}
