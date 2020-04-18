s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3)
{
    s32 sp24;
    s32 sp20;
    s32 sp1C;

    sp24 = arg0 + arg1;
    sp20 = arg1 + arg2;
    sp1C = 0;
    if ((((sp24 != 0) || (sp20 != 0)) || (sp20 = func_00400090(sp20), (sp20 != 0))) || (arg3 != 0))
    {
        sp1C = 1;
    }
    else
    {

    }
    return sp1C;
}
