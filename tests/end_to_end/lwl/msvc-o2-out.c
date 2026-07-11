extern s32 ??_C@_03LKLC@ghi?$AA@;
extern ? ??_C@_06CBKI@abcdef?$AA@;

void test(void) {
    s32 sp4;
    s16 sp8;
    s8 spA;

    sp4 = ??_C@_06CBKI@abcdef?$AA@.unk0;
    sp8 = ??_C@_06CBKI@abcdef?$AA@.unk4;
    spA = ??_C@_06CBKI@abcdef?$AA@.unk6;
    foo(&sp4);
    a1.unk1 = (s32) a2.unk1;
    a3->unk0 = (s32) a1.unk0;
    a3->data.ar[0] = a1.data.ar[0];
    *buf = ??_C@_03LKLC@ghi?$AA@;
}
