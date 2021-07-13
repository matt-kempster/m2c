? bar(); // extern

void test(void) {
    bar();
    if (4 == 0) {
        goto block_3;
    }
block_1:
    if (1 == 0) {
        goto block_3;
    }
    bar();
block_3:
    if ((1 < 1) < 0) {
        goto block_1;
    }
    return;
}
