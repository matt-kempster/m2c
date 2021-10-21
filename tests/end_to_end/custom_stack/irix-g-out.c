struct _mips2c_stack_test {
    char pad0[0x18];
    s8 sp18;                                        /* +0x18; inferred */
    char pad19[0x3];
    s32 sp1C;                                       /* +0x1C; inferred */
    s32 sp20;                                       /* +0x20; inferred */
    s8 sp24;                                        /* +0x24; inferred */
    char pad25[0x3];
    s8 sp28;                                        /* +0x28; inferred */
    char pad29[0x3];
    s8 sp2C;                                        /* +0x2C; inferred */
    char pad2D[0x2];
    s8 sp2F;                                        /* +0x2F; inferred */
};                                                  /* size 0x30 */

? func_00400090(s8 *);                              /* static */
s32 test(void *arg0);                               /* static */

s32 test(void *arg0) {
    s8 sp2F;
    s8 sp2C;
    s8 sp28;
    s8 sp24;
    s32 sp20;
    s32 sp1C;
    s8 sp18;

    func_00400090(&sp2F);
    func_00400090(&sp2C);
    func_00400090(&sp28);
    func_00400090(&sp24);
    func_00400090(&sp18);
    sp2F = arg0->unk0 + arg0->unk4;
    sp2C = arg0->unk0 + arg0->unk8;
    sp28 = arg0->unk4 + arg0->unk8;
    sp18 = arg0->unk0 * sp2F;
    sp1C = arg0->unk4 * (s16) sp2C;
    sp20 = arg0->unk8 * (s32) sp28;
    if (sp2F != 0) {
        sp24 = arg0;
    } else {
        sp24 = &sp18;
    }
    return sp2F + (s16) sp2C + (s32) sp28 + *(s32 *) sp24 + sp1C;
}
