extern s32 a;
extern ? b;
extern ? c;
extern ? *d;

void test(void) {
    a = 0x3F99999A;
    b.unk0 = 0x402A0000;
    b.unk4 = 0;
    c.unk0 = 0x420A13B8;
    c.unk4 = 0x60000000;
    d = "\"hello\"\n\x01";
}
