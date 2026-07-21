s32 test(s32 value) {
    u32 temp_r4;

    temp_r4 = value - 0xA;
    switch (temp_r4) {
    case 0:
        return 0xB;
    case 1:
        return 0x2A;
    case 2:
        return 0x13;
    case 3:
        return 0x49;
    default:
        return -7;
    }
}
