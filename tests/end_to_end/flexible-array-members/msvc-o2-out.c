s32 test(s32 arg0, s32 arg1, s32 arg2) {
    return *(arg0 + ((arg2 * 4) + 4)) + *(arg1 + ((arg2 * 8) + 8));
}
