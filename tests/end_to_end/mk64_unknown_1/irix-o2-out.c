s32 test(u32 arg0, s32 arg1, s32 arg2)
{
    s32 temp_a2;
    s32 temp_t0;
    s32 temp_v0;
    s32 temp_v0_2;
    s32 temp_v0_3;
    s32 temp_v1;
    s32 temp_v1_2;
    s32 phi_v0;
    void *phi_a0;
    s32 phi_v1;
    s32 phi_v0_2;
    void *phi_a0_2;
    s32 phi_v1_2;
    s32 phi_return;
    s32 phi_a2;
    s32 phi_a2_2;

    temp_v0 = ((0x80150000 + ((arg0 >> 0x18) * 4))->unk258 + (arg0 & 0xFFFFFF)) + 0x80000000;
    phi_return = temp_v0;
    temp_t0 = arg1;
    phi_v0_2 = temp_v0;
    phi_a2 = arg2;
    phi_v1_2 = 0;
    phi_v0 = temp_v0;
    phi_a0 = (arg2 * 4) + &D_8015F668;
    phi_v1 = 0;
    phi_a2_2 = arg2;
    do
    {
        *phi_a0 = phi_v0;
        temp_v1 = phi_v1 + 1;
        temp_v0_2 = phi_v0 + 0x10;
        temp_a2 = phi_a2_2 + 1;
        phi_v0 = temp_v0_2;
        phi_a0 = phi_a0 + 4;
        phi_v1 = temp_v1;
        phi_a2_2 = temp_a2;
    } while ((temp_t0 != temp_v1));
block_7:
    return phi_return;
}
