s64 test(s64 a, s64 b, u32 arg1, u32 arg3) {
    return (s64) (((u64) arg1 << 0x20) | (u32) a) / (s64) (((u64) arg3 << 0x20) | (u32) b);
}
