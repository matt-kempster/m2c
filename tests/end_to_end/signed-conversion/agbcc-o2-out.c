extern s32 glob;

void test(u8 arg0) {
    glob = (s32) arg0;
    glob = (s32) (u8) (arg0 * 2);
    glob = (s32) (u8) (arg0 * 3);
    glob = (s32) (s16) arg0;
    glob = (s32) (s16) (arg0 * 2);
    glob = (s32) (s16) (arg0 * 3);
}
