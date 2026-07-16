u32 test(u32 value) {
    return value * 2;
}

s32 test_shal(s32 value) {
    return value * 2;
}

u32 test_shlr(u32 value) {
    return value >> 1U;
}

s32 test_shar(s32 value) {
    return value >> 1;
}

u32 test_shll2(u32 value) {
    return value << 2U;
}

u32 test_shlr2(u32 value) {
    return value >> 2U;
}

u32 test_shll8(u32 value) {
    return value << 8U;
}

u32 test_shlr8(u32 value) {
    return value >> 8U;
}

u32 test_shll16(u32 value) {
    return value << 0x10U;
}

u32 test_shlr16(u32 value) {
    return value >> 0x10U;
}

u32 test_rotl(u32 value) {
    return (u32) (value << 1U) | (u32) (value >> 0x1FU);
}

u32 test_rotr(u32 value) {
    return (u32) (value >> 1U) | (u32) (value << 0x1FU);
}
