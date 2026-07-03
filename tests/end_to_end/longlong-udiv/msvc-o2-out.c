u64 test(u32 arg0, u32 arg1, u32 arg2, u32 arg3) {
    return (((u64) arg1 << 0x20) | arg0) / (((u64) arg3 << 0x20) | arg2);
}
