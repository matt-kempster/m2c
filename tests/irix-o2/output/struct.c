void *test(void *arg0)
{
    arg0->unk4 = (s32) (*arg0 + arg0->unk4);
    return arg0;
}
