s32 test(s32 arg0, s32 arg1, s32 arg2)
{
    s32 temp_v1;

    *sp = (?32) D_400120;
    temp_v1 = arg0 * 4;
    sp->unk4 = (s16) D_400120.unk4;
    return ((arg2 + temp_v1)->unk4 + (*(sp + arg0) * *(arg1 + temp_v1))) + *(&D_410130 + (arg0 * 2));
}
