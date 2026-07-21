s32 test(s32 lhs, s32 rhs) {
    return (s16) lhs * (s16) rhs;
}

u32 test_mulu(u32 lhs, u32 rhs) {
    return (u16) lhs * (u16) rhs;
}

s32 test_delay(s32 lhs, s32 old_rhs, s32 rhs) {
    return lhs / rhs;
}
