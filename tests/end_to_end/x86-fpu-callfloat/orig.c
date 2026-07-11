extern float mix(float u, float v);

float test(float a, float b) {
    return mix(a * b, a - b) + a;
}
