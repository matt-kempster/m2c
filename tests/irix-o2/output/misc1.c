void test(s32 arg0, s32 arg1)
{
    ?32 sp24;
    void *temp_v0;
    ? temp_ret;

    temp_v0 = (D_410170 + (arg0 * 8));
    sp24 = (?32) temp_v0->unk8;
    temp_ret = func_00400140(1, 2, (temp_v0->unk4 + 1), arg1, arg0);
    if (temp_ret != 0)
    {
        func_00400158(sp24, temp_ret, sp2C);
        *(&D_410178 + arg0) = (u8)5;
        return sp28;
    }
    return sp28;
}
