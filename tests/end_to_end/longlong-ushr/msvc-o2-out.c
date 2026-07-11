u64 test(u64 a, s32 b, u32 arg1) {
    return (((u64) arg1 << 0x20) | (u32) a) >> b;
}
