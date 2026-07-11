void test(s32 x) {
    glob = (s32) (s8) x;
    glob = (s32) (s8) ((s8) x * 2);
    glob = (s32) (s8) ((s8) x * 3);
    glob = (s32) (s16) x;
    glob = (s32) (s16) (x * 2);
    glob = (s32) (s16) (x * 3);
}
