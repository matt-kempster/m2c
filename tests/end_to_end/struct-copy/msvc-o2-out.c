extern ? _a;
extern ? _b;

void test(s32 arg0, s32 arg1) {
    M2C_MEMCPY(&_a, &_b, 0x64 * 4);
    M2C_MEMCPY(arg0, arg1, 0x19 * 4);
}
