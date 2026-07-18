u32 test(u32 lhs, u32 rhs) {
    return lhs & rhs;
}

u32 test_or(u32 lhs, u32 rhs) {
    return lhs | rhs;
}

u32 test_xor(u32 lhs, u32 rhs) {
    return lhs ^ rhs;
}

u32 test_not(u32 value) {
    return ~value;
}

s32 test_neg(s32 value) {
    return -value;
}

s32 test_eq(s32 lhs, s32 rhs) {
    return lhs == rhs;
}

s32 test_ge(s32 lhs, s32 rhs) {
    return lhs >= rhs;
}

s32 test_gt(s32 lhs, s32 rhs) {
    return lhs > rhs;
}

s32 test_hs(u32 lhs, u32 rhs) {
    return lhs >= rhs;
}

s32 test_hi(u32 lhs, u32 rhs) {
    return lhs > rhs;
}

s32 test_extsb(s8 value) {
    return (s32) value;
}

s32 test_extsw(s16 value) {
    return (s32) value;
}

u32 test_extub(u8 value) {
    return (u32) value;
}

u32 test_extuw(u16 value) {
    return (u32) value;
}

u32 test_swapw(u32 value) {
    return (value << 0x10U) | (value >> 0x10U);
}
