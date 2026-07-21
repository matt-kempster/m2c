u32 inner(?);                                       /* extern */
extern u32 g;

s64 test(void) {
    u32 temp_call_arg;

    temp_call_arg = g;
    return (s64) (((u64) 3U << 0x20) | inner(1)) * (s64) (((u64) temp_call_arg << 0x20) | 2U);
}

s32 dead(void) {
    inner(1);
    return 0;
}
