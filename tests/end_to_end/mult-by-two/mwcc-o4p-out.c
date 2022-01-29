s32 test(s32 arg0) {
    *NULL = (f64) (*NULL * *NULL);
    *NULL = (f32) ((f32) *NULL * (f32) *NULL);
    return arg0;
}
