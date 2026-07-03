extern s32 _global;

void test(s32 arg0, s32 arg1, s32 arg2) {
    _global = (s32) (u8) (arg0 == arg1);
    _global = (s32) (u8) (arg0 != arg2);
    _global = (s32) (u8) (arg0 < arg1);
    _global = (s32) (u8) (arg0 <= arg1);
    _global = (s32) (u8) (arg0 == 0);
    _global = (s32) (u8) (arg1 != 0);
    _global = (s32) (u8) (arg1 > 0);
    _global = (s32) (u8) (arg1 <= 0);
}
