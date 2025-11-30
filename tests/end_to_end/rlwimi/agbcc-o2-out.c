extern u8 x;
extern u8 y;

void test(void *arg0) {
    x = (((((-2 & x) | (arg0->unk0 & 1)) & ~2) | ((arg0->unk1 & 1) * 2)) & ~0x7C) | ((arg0->unk2 & 0x1F) * 4);
    x = (s16) ((0xFFFFF07F & (u16) x) | ((arg0->unk3 & 0x1F) << 7));
    y = (y | 1) & ~2;
}
