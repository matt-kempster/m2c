f32 test(void) {
    f32 temp_f0;

    temp_f0 = D_4100E0;
    D_4100E0 = (f32) (2.0f * temp_f0);
    D_4100E8 = (f64) (2.0 * D_4100E8);
    return temp_f0;
}
