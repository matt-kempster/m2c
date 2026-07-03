extern s32 _x;
extern s32 _y;

void test(void *arg0) {
    s32 temp_ecx;
    s32 temp_ecx_2;
    s32 temp_ecx_3;

    temp_ecx = (arg0->unk0 & 1) | (_x & ~1);
    _x = temp_ecx;
    temp_ecx_2 = (temp_ecx & ~2) | ((arg0->unk1 & 1) * 2);
    _x = temp_ecx_2;
    temp_ecx_3 = (temp_ecx_2 & ~0x7C) | ((arg0->unk2 & 0x1F) * 4);
    _x = temp_ecx_3;
    _x = ((arg0->unk3 & 0x1F) << 7) | (temp_ecx_3 & ~0xF80);
    _y = (_y & ~2) | 1;
}
