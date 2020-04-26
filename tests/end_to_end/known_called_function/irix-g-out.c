void test(s32 arg0, s32 arg1, ? arg2)
{
    s32 sp1C;

    sp1C = 0;
loop_1:
    sp1C = foo(sp1C, arg1);
    goto loop_1;
}
