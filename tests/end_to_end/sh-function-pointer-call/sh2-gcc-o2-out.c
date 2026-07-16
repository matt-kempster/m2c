s32 test(s32 (*callback)(s32), s32 value) {
    return callback(value) + 1;
}
