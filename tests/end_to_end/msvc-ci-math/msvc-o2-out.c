f64 test(f64 arg0, f64 arg2) {
    f64 temp_f0;

    temp_f0 = pow(arg0, arg2);
    return temp_f0 + temp_f0 + fmod(arg0, arg2);
}
