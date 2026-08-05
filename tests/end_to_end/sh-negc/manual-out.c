u32 test(u32 value) {
    return (u32) -(s32) value;
}

u32 test_set(u32 value) {
    return -(s32) value - 1;
}

u32 test_chain(u32 low, u32 high) {
    return -(s32) high - (low != 0);
}

u32 test_set_chain(u32 low, u32 high) {
    return -(s32) high - 1;
}
