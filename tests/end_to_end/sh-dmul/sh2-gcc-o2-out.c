s32 test(s32 lhs, s32 rhs) {
    return (s32) (((s64) lhs * (s64) rhs) >> 0x20);
}

u32 test_unsigned(u32 lhs, u32 rhs) {
    return (u32) (((u64) lhs * (u64) rhs) >> 0x20);
}
