s32 test(s32 *x) {
    s32 sp10;
    s32 *sp14;
    s32 *temp_edx;
    s32 temp_eax;
    s32 temp_ecx;

    temp_edx = x + 0x28;
    temp_eax = x->unk190;
    x = (s8) x->unk0;
    sp14 = temp_edx;
    sp10 = temp_eax;
    x = (s8) ((s8) x + *func_00400090(&x));
    sp14 = func_00400090(&sp14);
    temp_ecx = sp10 + *func_00400090(&sp10);
    sp10 = temp_ecx;
    return (s8) x + *sp14 + temp_ecx;
}
