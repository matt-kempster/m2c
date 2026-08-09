u32 test(u32 value) {
    return -value;
}

u32 test_set(u32 value) {
    return -value - 1;
}

u32 test_chain(u32 low, u32 high) {
    return -high - (low != 0);
}

u32 test_set_chain(u32 low, u32 high) {
    return -high - 1;
}
