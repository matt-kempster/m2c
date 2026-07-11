void test(s32 a, s32 b, s32 c) {
    _global = (s32) (u8) (a == b);
    _global = (s32) (u8) (a != c);
    _global = (s32) (u8) (a < b);
    _global = (s32) (u8) (a <= b);
    _global = (s32) (u8) (a == 0);
    _global = (s32) (u8) (b != 0);
    _global = (s32) (u8) (b > 0);
    _global = (s32) (u8) (b <= 0);
}
