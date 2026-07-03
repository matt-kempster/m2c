extern s8 _a;
extern ? _ar;
extern u8 _b;
extern s16 _c;
extern u16 _d;
extern s32 _e;
extern s32 _f;

void test(void) {
    _ar.unk0 = (s32) _a;
    _ar.unk4 = (s32) _b;
    _ar.unk8 = (s32) _c;
    _ar.unkC = (s32) _d;
    _ar.unk10 = (s32) _e;
    _ar.unk14 = (s32) _f;
}
