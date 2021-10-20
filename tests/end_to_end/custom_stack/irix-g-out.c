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
