s32 test(s32 *x) {
    u8 sp0;
    s32 *sp4;
    s32 sp8;
    s32 temp_r2;

    sp0 = x->unk0;
    sp4 = x + 0x28;
    sp8 = x->unk190;
    sp0 = *func_00400090(&sp0) + sp0;
    sp4 = func_00400090(&sp4);
    temp_r2 = sp8 + *func_00400090(&sp8);
    sp8 = temp_r2;
    return sp0 + *sp4 + temp_r2;
}
