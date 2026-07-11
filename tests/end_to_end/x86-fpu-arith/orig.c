extern float g_result;

void test(float a, float b, float c) {
    g_result = ((a + b) * c) - (a / b);
}
