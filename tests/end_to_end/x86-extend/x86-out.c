extern s8 _sbyte;
extern s16 _sword;
extern u8 _ubyte;
extern u16 _uword;

s32 test(void *arg0) {
    return _sbyte + _ubyte + _sword + _uword + arg0->unk1;
}
