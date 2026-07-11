void test(s32 x) {
    _glob = (s32) (s8) x;
    _glob = (s32) (s8) ((s8) x * 2);
    _glob = (s32) (s8) ((s8) x * 3);
    _glob = (s32) (s16) x;
    _glob = (s32) (s16) (x * 2);
    _glob = (s32) (s16) (x * 3);
}
