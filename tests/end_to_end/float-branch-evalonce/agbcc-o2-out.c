s32 __ltsf2(s32, s32);                              /* extern */
extern s32 x;

void test(void) {
    s32 temp_r1;
    s32 temp_r1_2;

    temp_r1 = x;
    x = 0x40A00000;
    if (__ltsf2(temp_r1, 0) < 0) {
        x = 0x40C00000;
    }
    temp_r1_2 = x;
    x = 0x40400000;
    if (__ltsf2(temp_r1_2, 0) >= 0) {
        x = 0x40E00000;
    }
}
