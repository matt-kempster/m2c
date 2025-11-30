? memcpy(? *, s32, s32);                            /* extern */
extern ? a;

void test(? *arg0, s32 arg1) {
    memcpy(&a, .L3.unk4, 0x190);
    memcpy(arg0, arg1, 0x64);
}
