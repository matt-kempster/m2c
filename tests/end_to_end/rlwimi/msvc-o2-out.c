s32 test(s8 *ptr) {
    s32 temp_eax;
    s32 temp_ecx;
    s32 temp_ecx_2;
    s32 temp_ecx_3;

    temp_ecx = (ptr->unk0 & 1) | (_x & ~1);
    _x = temp_ecx;
    temp_ecx_2 = (temp_ecx & ~2) | ((ptr->unk1 & 1) * 2);
    _x = temp_ecx_2;
    temp_ecx_3 = (temp_ecx_2 & ~0x7C) | ((ptr->unk2 & 0x1F) * 4);
    _x = temp_ecx_3;
    temp_eax = ((ptr->unk3 & 0x1F) << 7) | (temp_ecx_3 & ~0xF80);
    _x = temp_eax;
    _y = (_y & ~2) | 1;
    return temp_eax;
}
