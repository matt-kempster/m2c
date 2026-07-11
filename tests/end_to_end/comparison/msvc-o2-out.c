void test(s32 a, s32 b, s32 c) {
    global = (s32) (u8) (a == b);
    global = (s32) (u8) (a != c);
    global = (s32) (u8) (a < b);
    global = (s32) (u8) (a <= b);
    global = (s32) (u8) (a == 0);
    global = (s32) (u8) (b != 0);
    global = (s32) (u8) (b > 0);
    global = (s32) (u8) (b <= 0);
}
