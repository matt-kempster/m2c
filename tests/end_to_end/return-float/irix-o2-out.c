f32 test(f32 arg0); // static

f32 test(f32 arg0) {
    if (arg0 != 0.0f) {
        return 15.0f;
    }
    return arg0;
}
