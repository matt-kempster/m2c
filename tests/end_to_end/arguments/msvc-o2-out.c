extern f32 _globalf;
extern s32 _globali;

void test(f32 arg0, s32 arg1, f32 arg2, s32 arg3, f32 arg4, s32 arg5) {
    _globali = arg1 + arg3 + arg5;
    _globalf = arg0 + arg2 + arg4;
}
