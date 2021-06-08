void test(void) {
    ?32 sp18;

    sp18.unk0 = (?32) D_400170.unk0;
    sp18.unk4 = (first 3 bytes) D_400170.unk4;
    func_004000B0(&sp18);
    D_410181 = (unaligned s32) D_410189;
    D_410190.unk0 = (unaligned s32) D_410180.unk0;
    D_410190.unk4 = (u8) D_410180.unk4;
    D_410198 = (?32) (unaligned s32) D_400178;
}
