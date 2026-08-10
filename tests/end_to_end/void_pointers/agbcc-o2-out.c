s32 test(s32 *x) {
    s32 *sp4;
    s32 sp8;
    s32 temp_r2;

    subroutine_arg0 = x->unk0;
    sp4 = x + 0x28;
    sp8 = x->unk190;
    subroutine_arg0 = (u8) (*func_00400090(&subroutine_arg0) + subroutine_arg0);
    sp4 = func_00400090(&sp4);
    temp_r2 = sp8 + *func_00400090(&sp8);
    sp8 = temp_r2;
    return subroutine_arg0 + *sp4 + temp_r2;
}
