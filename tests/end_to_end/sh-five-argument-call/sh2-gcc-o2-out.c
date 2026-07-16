s32 test(s32 value) {
    return callee(value, value + 1, value + 2, value + 3, value + 4) + 5;
}
