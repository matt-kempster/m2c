extern s8 sbyte;
extern s16 sword;
extern u8 ubyte;
extern u16 uword;

s32 test(void *arg0) {
    return sbyte + ubyte + sword + uword + arg0->unk1;
}
