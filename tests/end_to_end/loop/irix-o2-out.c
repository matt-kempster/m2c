s32 test(void *arg0, s32 arg1)
{
    s32 temp_a3;
    s32 temp_v0;
    s32 temp_v0_2;
    void *temp_v1;
    void *phi_v1;
    s32 phi_v0;
    void *phi_v1_2;
    s32 phi_v0_2;
    s32 phi_return;
    s32 phi_v0_3;

    phi_return = 0;
    temp_a3 = arg1;
    phi_v0_3 = 0;
    phi_v1 = arg0;
    phi_v0 = 0;
    do
    {
        temp_v0 = phi_v0 + 1;
        *phi_v1 = (u8)0;
        phi_v1 = phi_v1 + 1;
        phi_v0 = temp_v0;
    } while ((temp_a3 != temp_v0));
block_7:
    return phi_return;
}
