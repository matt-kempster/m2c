extern f32 globalf;
extern s32 globali;

f32 test(s32 arg0, s32 arg1, f32 arg8, f64 arg9) {
    s32 sp18;
    s32 sp1C;
    s32 sp20;
    s32 sp24;
    f64 sp28;

    sp1C = arg1 + 3;
    sp24 = arg0 ^ 0x80000000;
    sp20 = 0x43300000;
    sp18 = 0x43300000;
    sp28 = (bitwise f64) (s32) arg8;
    globali = unksp2C;
    globalf = (f32) (bitwise f64) sp20 - (f32) 4503601774854144.0;
    return ((f32) (bitwise f64) sp18 - (f32) 4503599627370496.0) + (f32) ((f64) (f32) (arg9 + 5.0) + 5.3);
}
