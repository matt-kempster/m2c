extern f32 globalf;
extern s32 globali;

s32 test(s32 arg0, s32 arg1, f32 arg8, f64 arg9) {
    globali = (s32) arg8;
    globalf = (f32) arg0;
    return arg0 ^ 0x80000000;
}
