s32 test(void)
{
    s32 temp_ret;
    s32 phi_return;

    temp_ret = func_0040010C(1);
    if (temp_ret != 0)
    {
        phi_return = temp_ret & 0xFFFF;
    }
    else
    {
        if (D_410120 != 0x7B)
        {
            return func_0040010C(2);
        }
        phi_return = func_0040010C(3);
    }
    return phi_return;
}
