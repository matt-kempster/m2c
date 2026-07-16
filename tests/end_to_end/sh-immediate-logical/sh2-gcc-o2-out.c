u32 test(u32 value) {
    return value & 0x5A;
}

u32 test_or(u32 value) {
    return value | 0x5A;
}

u32 test_xor(u32 value) {
    return value ^ 0x5A;
}

s32 test_tst(u32 value) {
    return (value & 0x5A) == 0;
}
