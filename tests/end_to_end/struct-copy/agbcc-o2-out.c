? memcpy(? *, ? *, s32);                            /* extern */
extern ? a;
extern ? b;

void test(? *arg0, ? *arg1) {
    memcpy(&a, &b, 0x190);
    memcpy(arg0, arg1, 0x64);
}
