s64 test(u32 arg0, u32 arg1, s32 arg2) {
    return (s64) (((u64) arg1 << 0x20) | arg0) >> arg2;
}
