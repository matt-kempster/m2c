s32 __ltsf2(s32, s32);                              /* extern */
extern s32 x;

void test(void) {
    s32 temp_r1;
    s32 temp_r1_2;
    s32 temp_r5;

    temp_r1 = x;
    x = .L5.unk4;
    temp_r5 = .L5.unk8;
    if (__ltsf2(temp_r1, temp_r5) < 0) {
        x = .L5.unkC;
    }
    temp_r1_2 = x;
    x = .L5.unk10;
    if (__ltsf2(temp_r1_2, temp_r5) >= 0) {
        x = .L5.unk14;
    }
}
