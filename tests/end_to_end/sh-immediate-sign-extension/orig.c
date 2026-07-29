signed test(void) {
    return -1;
}

signed test_min(void) {
    return -128;
}

signed test_max(void) {
    return 127;
}

signed test_add(signed value) {
    return value - 1;
}

int test_compare(signed value) {
    return value == -1;
}
