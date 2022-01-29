? strcpy(? *, ?, s32);                              /* extern */
? foo(s32 *, u16);                                  /* static */
static ? buf;

void test(void) {
    s32 sp8;
    u16 spC;
    u8 spE;
    s32 *temp_r3;
    s32 temp_r5;
    u16 temp_r4;

    temp_r3 = &sp8;
    temp_r4 = (u16) *NULL;
    sp8 = *NULL;
    spC = temp_r4;
    spE = (u8) *NULL;
    foo(temp_r3, temp_r4);
    *(void *)1 = (s32) *(s32 *)1;
    temp_r5 = *NULL;
    *NULL = temp_r5;
    *NULL = (u8) *NULL;
    strcpy(&buf, 0, temp_r5);
}
