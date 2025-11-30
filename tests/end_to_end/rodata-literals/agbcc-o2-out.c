extern s32 a;
extern ? b;
extern ? c;
extern ? d;

void test(void) {
    ? *temp_r2;

    a = 0x3F99999A;
    b.unk0 = &c;
    b.unk4 = &d;
    temp_r2 = "\"hello\"\n\x01";
    temp_r2->unk0 = (s32) .L3.unk18;
    temp_r2->unk4 = (s32) .L3.unk1C;
    *.L3.unk20 = .L3.unk24;
}
