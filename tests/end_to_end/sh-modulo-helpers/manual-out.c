s32 test(s32 lhs, s32 rhs) {
    return (lhs % rhs) + 1;
}

s32 mod_extu_delay(s32 lhs, s32 rhs) {
    return lhs % (u16) rhs;
}

s32 mod_split_dividend(s32 lhs, s32 rhs) {
    return lhs % (u16) rhs;
}

s32 mod_feeds_multiply(s32 lhs, s32 rhs) {
    return (lhs % (u16) rhs) * lhs;
}

u32 mod_moved_quotient(u32 lhs, u32 rhs) {
    return (lhs % rhs) + 1;
}

u32 mod_interleaved(u32 lhs, u32 rhs) {
    return lhs % rhs;
}
