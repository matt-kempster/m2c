s32 _sink(s32);                                     /* extern */

s32 test(s32 arg0) {
    u32 temp_r0;

    temp_r0 = sink() & 0x7F;
    switch (temp_r0) {                              /* irregular */
    case 0:
        return 0xB;
    case 1:
        return 0x30;
    case 2:
        return 0x55;
    case 3:
        return 0x7A;
    case 4:
        return 0x20;
    case 5:
        return 0x45;
    case 6:
        return 0x6A;
    case 7:
        return 0x10;
    case 8:
        return 0x35;
    case 9:
        return 0x5A;
    case 10:
        return 0;
    case 11:
        return 0x25;
    case 12:
        return 0x4A;
    case 13:
        return 0x6F;
    case 14:
        return 0x15;
    case 15:
        return 0x3A;
    case 16:
        return 0x5F;
    case 17:
        return 5;
    case 18:
        return 0x2A;
    case 19:
        return 0x4F;
    case 20:
        return 0x74;
    case 21:
        return 0x1A;
    case 22:
        return 0x3F;
    case 23:
        return 0x64;
    case 24:
        return 0xA;
    case 25:
        return 0x2F;
    case 26:
        return 0x54;
    case 27:
        return 0x79;
    case 28:
        return 0x1F;
    case 29:
        return 0x44;
    case 30:
        return 0x69;
    case 31:
        return 0xF;
    case 32:
        return 0x34;
    case 33:
        return 0x59;
    case 34:
        return 0x7E;
    case 35:
        return 0x24;
    case 36:
        return 0x49;
    case 37:
        return 0x6E;
    case 38:
        return 0x14;
    case 39:
        return 0x39;
    case 40:
        return 0x5E;
    case 41:
        return 4;
    case 42:
        return 0x29;
    case 43:
        return 0x4E;
    default:
        return sink(temp_r0 + 1);
    }
}
