s32 __mulsf3(s32, s32);                             /* extern */
s32 __subsf3(s32, s32);                             /* extern */

void test(s32 arg0) {
    s32 temp_r5;

    temp_r5 = .L3.unk4 - (arg0 >> 1);
    __mulsf3(temp_r5, __subsf3(.L3.unk8, __mulsf3(__mulsf3(__mulsf3(arg0, 0x3F000000), temp_r5), temp_r5)));
}
