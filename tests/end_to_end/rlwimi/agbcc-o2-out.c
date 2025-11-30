extern u8 x;

void test(void *arg0) {
    x = (((((-2 & x) | (arg0->unk0 & 1)) & ~2) | ((arg0->unk1 & 1) * 2)) & ~0x7C) | ((arg0->unk2 & 0x1F) * 4);
    x = (s16) ((.L3.unk4 & (u16) x) | ((arg0->unk3 & 0x1F) << 7));
    *.L3.unk8 = (*.L3.unk8 | 1) & ~2;
}
