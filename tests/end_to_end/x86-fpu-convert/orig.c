extern int g_i;

void test(int n, float scale) {
    g_i = (int) ((float) n * scale);
}
