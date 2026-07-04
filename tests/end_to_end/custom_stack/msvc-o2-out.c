struct _m2c_stack_test {
    /* 0x00 */ char pad0[0x14];
    /* 0x14 */ s32 sp14;                            /* inferred */
    /* 0x18 */ s32 sp18;                            /* inferred */
    /* 0x1C */ s32 sp1C;                            /* inferred */
    /* 0x20 */ s32 sp20;                            /* inferred */
    /* 0x24 */ char pad24[3];
    /* 0x27 */ s32 sp27;                            /* inferred */
    /* 0x28 */ s32 sp28;                            /* overlap; inferred */
    /* 0x2C */ s32 sp2C;                            /* inferred */
    /* 0x30 */ s32 sp30;                            /* inferred */
    /* 0x34 */ s32 sp34;                            /* inferred */
    /* 0x38 */ s32 sp38;                            /* inferred */
    /* 0x3C */ s32 sp3C;                            /* inferred */
};                                                  /* size = 0x40 */

? frob(s32 *);                                      /* static */

void *test(s32 *arg0) {
    s32 sp27;
    s32 sp28;
    s32 sp2C;
    s32 sp30;
    s32 sp34;
    s32 sp38;
    s32 sp3C;
    s32 *var_eax;
    s32 temp_ebx;
    s32 temp_edi;
    s32 temp_edx;
    s32 temp_esi;
    s32 temp_esi_2;
    s8 temp_ecx;

    frob(&sp27);
    frob(&sp2C);
    frob(&sp30);
    frob(&sp28);
    frob(&sp34);
    var_eax = arg0;
    temp_ecx = (s8) var_eax->unk0 + var_eax->unk4;
    arg0 = (s32 *) temp_ecx;
    temp_esi = var_eax->unk8 + (s16) var_eax->unk0;
    temp_edx = (s32) var_eax->unk8;
    temp_edi = (s32) var_eax->unk4;
    sp34 = temp_ecx * var_eax->unk0;
    sp2C = temp_esi;
    temp_ebx = temp_edi + temp_edx;
    temp_esi_2 = (s16) temp_esi * temp_edi;
    sp27 = temp_ecx;
    sp30 = temp_ebx;
    sp38 = temp_esi_2;
    sp3C = temp_edx * temp_ebx;
    if (temp_ecx == 0) {
        var_eax = &sp34;
    }
    sp28 = (s32) var_eax;
    return *var_eax + (s16) temp_esi + arg0 + temp_esi_2 + temp_ebx;
}
