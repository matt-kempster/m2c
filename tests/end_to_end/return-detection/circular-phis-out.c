? bar(); // extern

void test(void) {
    s32 phi_v0;

    bar();
    phi_v0 = 4;
    do {
        if (1 != 0) {
            bar();
            phi_v0 = 5;
        }
    } while ((phi_v0 > 1) < 0);
}
