u32 test(u32 lhs, u32 rhs) {
    return lhs - rhs;
}

u32 test_set(u32 lhs, u32 rhs) {
    return (lhs - rhs) - 1;
}

u32 test_chain(u32 lhs_low, u32 lhs_high, u32 rhs_low, u32 rhs_high) {
    return (lhs_high - rhs_high) - (lhs_low < rhs_low);
}
