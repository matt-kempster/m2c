u32 test(u32 value) {
    return value << 1;
}

u32 test_set(u32 value) {
    return (value << 1) | 1;
}

u32 test_chain(u32 low, u32 high) {
    return (high << 1) | M2C_CARRY((low << 1));
}

u32 test_rotate_low(u32 low, u32 high) {
    return (low << 1) | (high >> 0x1F);
}
