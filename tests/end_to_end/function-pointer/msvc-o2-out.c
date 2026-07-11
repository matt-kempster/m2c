s32 test(s32 (*fn)(s32), s32 x) {
    return fn(x) + 1;
}
