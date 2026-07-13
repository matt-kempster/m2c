struct _m2c_stack_test {
    /* 0x00 */ s8 pad0[0x27];
    /* 0x27 */ s8 a;
    /* 0x28 */ struct Vec *d;
    /* 0x2C */ s16 b;
    /* 0x2E */ s8 pad1[2];
    /* 0x30 */ s32 c;
    /* 0x34 */ struct Vec e;
};                                                  /* size = 0x40 */

s32 test(struct Vec *v) {
    s8 a;
    struct Vec *d;
    s16 b;
    s32 c;
    struct Vec e;
    s32 temp_ebx;
    s32 temp_edi;
    s32 temp_edx;
    s32 temp_esi;
    s32 temp_esi_2;
    s8 temp_ecx;
    struct Vec *var_eax;

    frob(&a);
    frob(&b);
    frob(&c);
    frob(&d);
    frob(&e);
    var_eax = v;
    temp_ecx = var_eax->unk0 + var_eax->unk4;
    v = (struct Vec *) temp_ecx;
    temp_esi = var_eax->unk8 + (s16) var_eax->unk0;
    temp_edx = var_eax->z;
    temp_edi = var_eax->y;
    e = temp_ecx * var_eax->x;
    b = temp_esi;
    temp_ebx = temp_edi + temp_edx;
    temp_esi_2 = (s16) temp_esi * temp_edi;
    a = temp_ecx;
    c = temp_ebx;
    e.y = temp_esi_2;
    e.z = temp_edx * temp_ebx;
    if (temp_ecx == 0) {
        var_eax = &e;
    }
    d = var_eax;
    return (s32) (var_eax->x + (s16) temp_esi + v + temp_esi_2 + temp_ebx);
}
