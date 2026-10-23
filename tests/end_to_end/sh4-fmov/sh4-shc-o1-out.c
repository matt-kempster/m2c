extern f32 _g;

f32 test(void) {
    g = 1.0f;
    return 1.0f;
}

f32 load(void) {
    return g;
}
