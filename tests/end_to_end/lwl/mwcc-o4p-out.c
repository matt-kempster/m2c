? strcpy(? *, ? *, s32);                            /* extern */
? foo(s32 *, u16);                                  /* static */
extern ? $$24;
extern ? a1;
extern ? a2;
extern s32 a3;
static ? buf;
static s32 $$22 = 0x61626364;                       /* const */

void test(void) {
    s32 sp8;
    u16 spC;
    u8 spE;
    s32 *temp_r3;
    s32 temp_r5;
    u16 temp_r4;

    temp_r3 = &sp8;
    temp_r4 = (u16) $$22;
    sp8 = $$22;
    spC = temp_r4;
    spE = (u8) $$22;
    foo(temp_r3, temp_r4);
    a1.unk1 = (s32) a2.unk1;
    temp_r5 = a1.unk0;
    a3 = temp_r5;
    a3 = (u8) a1.unk0;
    strcpy(&buf, &$$24, temp_r5);
}
