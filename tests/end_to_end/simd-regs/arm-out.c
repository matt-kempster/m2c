void test(f32 *arg0, f32 *arg1) {
    *arg1 = *arg0;
    *arg1 = (f64) *arg0;
    *arg1 = (s128) *arg0;
}
