void test(s32 arg0)
{
    if (arg0 != 7)
    {
        func_004000F0(4, arg0);
        return;
    }
    func_004000F0(1, arg0);
    if (arg0 == 8)
    {
        func_004000F0(2, arg0);
    }
    func_004000F0(3);
}
