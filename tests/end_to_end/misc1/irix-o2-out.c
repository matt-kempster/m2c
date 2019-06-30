s32 test(s32 arg0, s32 arg1)
{
    s32 sp2C;
    s32 sp28;
    ?32 sp24;
    s32 temp_a1;
    s32 temp_a2;
    s32 temp_ret;
    void *temp_v0;

    temp_v0 = D_410170 + (arg0 * 8);
    temp_a2 = temp_v0->unk4 + 1;
    sp2C = temp_a2;
    sp24 = (?32) temp_v0->unk8;
    temp_ret = func_00400140(1, 2, temp_a2, arg1, arg0);
    temp_a1 = temp_ret;
    if (temp_ret == 0)
    {
        return 0;
    }
    sp28 = temp_a1;
    func_00400158(sp24, temp_a1, temp_a2);
    *(&D_410178 + arg0) = (u8)5;
    return sp28;
}
