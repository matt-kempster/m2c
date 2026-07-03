extern ? ??_C@_05DLON@hello?$AA@;
extern ? _globalArray;

s32 test(s32 arg0, s32 arg1, s32 arg2) {
    s32 sp4;
    s16 sp8;

    sp8 = ??_C@_05DLON@hello?$AA@.unk4;
    sp4 = ??_C@_05DLON@hello?$AA@.unk0;
    return (*(arg1 + (arg0 * 4)) * *(sp + (arg0 + 4))) + *(arg2 + ((arg0 * 4) + 4)) + *((arg0 * 2) + &_globalArray);
}
