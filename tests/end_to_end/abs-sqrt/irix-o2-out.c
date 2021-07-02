f32 test(f32 arg0); // static

f32 test(f32 arg0) {
    return (f32) sqrt(fabs((f64) sqrtf(fabsf(arg0))));
}
