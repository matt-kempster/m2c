u32 test(u32 value) {
    return value * 2;
}

u32 test_set(u32 value) {
    return (u32) (value * 2) | 1U;
}

u32 test_chain(u32 low, u32 high) {
    return (u32) (high * 2) | (u32) (low >> 0x1FU);
}
