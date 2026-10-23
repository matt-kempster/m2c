extern float g;

float test(void) {
    g = 1.0f;
    return g;
}

float load(void) {
    return g;
}
