void test(s32 arg0, s32 arg1)
{
    s32 sp4;
    s32 temp_t9;

    sp4 = 0;
    if (arg1 > 0)
    {
loop_1:
        *(arg0 + sp4) = (u8)0;
        temp_t9 = sp4 + 1;
        sp4 = temp_t9;
        if (temp_t9 < arg1)
        {
            goto loop_1;
        }
    }
}
