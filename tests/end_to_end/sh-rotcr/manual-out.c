u32 test(u32 value) {
    return value >> 1;
}

u32 test_set(u32 value) {
    return (value >> 1) | (1U << 0x1FU);
}

u32 test_chain(u32 low, u32 high) {
    return (low >> 1) | ((u32) (high & 1) << 0x1FU);
}

u32 test_rotate_high(u32 low, u32 high) {
    return (high >> 1) | ((u32) (low & 1) << 0x1FU);
}
