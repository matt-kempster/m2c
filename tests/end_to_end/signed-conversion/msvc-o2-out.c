extern s32 _glob;

void test(s32 arg0) {
    _glob = (s32) (s8) arg0;
    _glob = (s32) (s8) ((s8) arg0 * 2);
    _glob = (s32) (s8) ((s8) arg0 * 3);
    _glob = (s32) (s16) arg0;
    _glob = (s32) (s16) (arg0 * 2);
    _glob = (s32) (s16) (arg0 * 3);
}
