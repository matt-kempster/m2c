? foo(s32 *);                                       /* static */
extern s32 ??_C@_03LKLC@ghi?$AA@;
extern ? ??_C@_06CBKI@abcdef?$AA@;
extern ? _a1;
extern ? _a2;
extern ? _a3;
extern s32 _buf;

void test(void) {
    s32 sp4;
    s16 sp8;
    s8 spA;

    sp4 = ??_C@_06CBKI@abcdef?$AA@.unk0;
    sp8 = ??_C@_06CBKI@abcdef?$AA@.unk4;
    spA = ??_C@_06CBKI@abcdef?$AA@.unk6;
    foo(&sp4);
    _a1.unk1 = (s32) _a2.unk1;
    _a3.unk0 = (s32) _a1.unk0;
    _a3.unk4 = (s8) _a1.unk4;
    _buf = ??_C@_03LKLC@ghi?$AA@;
}
