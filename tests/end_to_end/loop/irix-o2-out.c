s32 test(void *arg0, s32 arg1)
{
    s32 temp_v0;
    void *phi_v1;
    s32 phi_v0;
    s32 phi_return;

    phi_return = 0;
    if (arg1 > 0)
    {
        phi_v1 = arg0;
        phi_v0 = 0;
loop_3:
        temp_v0 = phi_v0 + 1;
        *phi_v1 = (u8)0;
        phi_v1 = phi_v1 + 1;
        phi_v0 = temp_v0;
        phi_return = temp_v0;
        if (arg1 != temp_v0)
        {
            goto loop_3;
        }
    }
    return phi_return;
}
