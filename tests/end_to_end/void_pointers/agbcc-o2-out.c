s32 test(s32 *x, s32 arg4) {
    s32 temp_r2;

    unksp0 = x->unk0;
    unksp4 = x + 0x28;
    arg4 = x->unk190;
    unksp0 = (u8) (*func_00400090(&unksp0) + unksp0);
    unksp4 = func_00400090(&unksp4);
    temp_r2 = arg4 + *func_00400090(&arg4);
    arg4 = temp_r2;
    return unksp0 + *unksp4 + temp_r2;
}
