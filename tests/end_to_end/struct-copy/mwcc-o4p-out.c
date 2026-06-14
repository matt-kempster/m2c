static ? a;
static ? b;

void test(s32 arg0, s32 arg1) {
    M2C_STRUCT_COPY(&a, &b, 0x190);
    M2C_STRUCT_COPY(arg0, arg1, 0x64);
}
