s32 test(s32 arg0) {
    s32 sp4;

    sp4 = 0;
    if (sp4 < arg0)
    {
        loop_1:
        if (sp4 == 5)
        {
            sp4 = (s32) (sp4 * 2);
        }
        else
        {
            sp4 = (s32) (sp4 + 4);
        }
        if (sp4 < arg0)
        {
            goto loop_1;
        }
    }
    return;
    // (possible return value: sp4)
}
