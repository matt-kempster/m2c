s32 test(s32 *x) {
    s8 sp27;
    s32 *sp20;
    s32 sp1C;
    s32 temp_t4;

    sp20 = &x[0xA];
    sp27 = (s8) x->unk0;
    sp1C = x[0x64];
    sp27 += *func_00400090(&sp27);
    sp20 = func_00400090(&sp20);
    temp_t4 = sp1C + *func_00400090(&sp1C);
    sp1C = temp_t4;
    return *sp20 + sp27 + temp_t4;
}
