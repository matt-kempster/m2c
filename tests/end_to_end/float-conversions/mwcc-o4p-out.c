extern f64 dbl;
extern f32 flt;
extern u32 u;

? test(void) {
    u = (u32) flt;
    u = (u32) dbl;
    dbl = (f64) u;
    flt = (f32) u;
    return 0x43300000;
}
