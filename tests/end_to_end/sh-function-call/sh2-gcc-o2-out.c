s32 test(s32 value) {
    return callee(value, value + 1) + global;
}
