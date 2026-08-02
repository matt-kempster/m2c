extern int sink(int);

int test(int value) {
    value = sink(value) & 0x7F;
    switch (value) {
    case 0: return 11;
    case 1: return 48;
    case 2: return 85;
    case 3: return 122;
    case 4: return 32;
    case 5: return 69;
    case 6: return 106;
    case 7: return 16;
    case 8: return 53;
    case 9: return 90;
    case 10: return 0;
    case 11: return 37;
    case 12: return 74;
    case 13: return 111;
    case 14: return 21;
    case 15: return 58;
    case 16: return 95;
    case 17: return 5;
    case 18: return 42;
    case 19: return 79;
    case 20: return 116;
    case 21: return 26;
    case 22: return 63;
    case 23: return 100;
    case 24: return 10;
    case 25: return 47;
    case 26: return 84;
    case 27: return 121;
    case 28: return 31;
    case 29: return 68;
    case 30: return 105;
    case 31: return 15;
    case 32: return 52;
    case 33: return 89;
    case 34: return 126;
    case 35: return 36;
    case 36: return 73;
    case 37: return 110;
    case 38: return 20;
    case 39: return 57;
    case 40: return 94;
    case 41: return 4;
    case 42: return 41;
    case 43: return 78;
    default: return sink(value + 1);
    }
}
