s32 test(void)
{
    s32 temp_ret;

    temp_ret = func_0040010C(1);
    if (temp_ret != 0)
    {
        return temp_ret & 0xFFFF;
    }
    if (D_410120 != 0x7B)
    {
        return func_0040010C(2);
    }
    return func_0040010C(3);
}
