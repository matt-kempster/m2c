s32 func_00400140(MIPS2C_UNK, MIPS2C_UNK, s32, MIPS2C_UNK, s32); /* static */
MIPS2C_UNK func_00400158(s32, s32, s32);            /* static */
s32 test(s32 arg0, MIPS2C_UNK arg1);                /* static */
extern s32 D_410170;
extern MIPS2C_UNK D_410178;

s32 test(s32 arg0, MIPS2C_UNK arg1) {
    s32 sp2C;
    s32 sp28;
    s32 sp24;
    s32 temp_a2;
    s32 temp_v0_2;
    void *temp_v0;

    temp_v0 = D_410170 + (arg0 * 8);
    temp_a2 = MIPS2C_FIELD(temp_v0, s32 *, 0x4) + 1;
    sp2C = temp_a2;
    sp24 = MIPS2C_FIELD(temp_v0, s32 *, 0x8);
    temp_v0_2 = func_00400140(1, 2, temp_a2, arg1, arg0);
    if (temp_v0_2 == 0) {
        return 0;
    }
    sp28 = temp_v0_2;
    func_00400158(sp24, temp_v0_2, temp_a2);
    *(&D_410178 + arg0) = 5;
    return sp28;
}
