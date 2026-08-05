u32 test(u32 value) {
    return value >> 1;
}

u32 test_set(u32 value) {
    return (value >> 1) | (1U << 0x1FU);
}

u32 test_chain(u32 low, u32 high) {
    return (low >> 1) | ((u32) M2C_CARRY((high >> 1)) << 0x1FU);
}
