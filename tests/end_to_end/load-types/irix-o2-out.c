?32 *test(); // static
extern s8 D_410110;
extern u8 D_410111;
extern s16 D_410112;
extern u16 D_410114;
extern ?32 D_410118;
extern ?32 D_41011C;
extern ?32 D_410120;

?32 *test(void) {
    D_410120.unk0 = (?32) D_410110;
    D_410120.unk4 = (?32) D_410111;
    D_410120.unk8 = (?32) D_410112;
    D_410120.unkC = (?32) D_410114;
    D_410120.unk10 = (?32) D_410118;
    D_410120.unk14 = (?32) D_41011C;
    return &D_410120;
}
