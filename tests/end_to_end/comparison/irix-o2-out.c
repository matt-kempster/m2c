void *test(s32 arg0, s32 arg1, s32 arg2)
{
    D_410100 = (u32) ((u32) (arg0 ^ arg1) < 1U);
    D_410100 = (u32) (0U < (u32) (arg0 ^ arg2));
    D_410100 = (u32) (arg0 < arg1);
    D_410100 = (u32) ((arg1 < arg0) ^ 1);
    D_410100 = (u32) (arg0 < 1U);
    D_410100 = (u32) (0U < arg1);
    return &D_410100;
}
