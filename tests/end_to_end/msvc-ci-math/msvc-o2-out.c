f64 test(f64 a, f64 b) {
    f64 temp_f0;

    temp_f0 = pow(a, b);
    return temp_f0 + temp_f0 + fmod(a, b);
}
