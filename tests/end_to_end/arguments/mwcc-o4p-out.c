extern f32 globalf;
extern s32 globali;

void test(s32 arg0, s32 arg1, s32 arg2, f32 farg0, f32 farg1, f32 farg2) {
    globali = arg0 + (arg1 + arg2);
    globalf = farg2 + (farg0 + farg1);
}
