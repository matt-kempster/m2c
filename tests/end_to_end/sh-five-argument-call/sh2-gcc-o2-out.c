s32 test(s32 value) {
    return callee(value, value + 1, value + 2, value + 3, value + 4) + 5;
}

s32 test6(s32 value) {
    return callee6(value, value + 1, value + 2, value + 3, value + 4, value + 5);
}

s32 test7(s32 value) {
    return callee7(value, value + 1, value + 2, value + 3, value + 4, value + 5, value + 6);
}

s32 test8(s32 value) {
    return callee8(value, value + 1, value + 2, value + 3, value + 4, value + 5, value + 6, value + 7);
}
