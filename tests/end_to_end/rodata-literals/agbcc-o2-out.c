extern s32 a;

void test(void) {
    a = .L3.unk4;
    .L3.unk8->unk0 = (s32) .L3.unkC;
    .L3.unk8->unk4 = (s32) .L3.unk10;
    .L3.unk14->unk0 = (s32) .L3.unk18;
    .L3.unk14->unk4 = (s32) .L3.unk1C;
    *.L3.unk20 = .L3.unk24;
}
