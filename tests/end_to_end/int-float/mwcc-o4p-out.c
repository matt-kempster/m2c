extern f32 globalf;
extern s32 globali;

f32 test(s32 arg0, s32 arg1, f32 farg0, f32 farg1) {
    globali = (s32) farg0;
    globalf = (f32) arg0;
    return (f32) (arg1 + 3) + (farg1 + 5.0f);
}
