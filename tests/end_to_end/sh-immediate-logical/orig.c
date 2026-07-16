unsigned test(unsigned value) {
    return value & 0x5A;
}

unsigned test_or(unsigned value) {
    return value | 0x5A;
}

unsigned test_xor(unsigned value) {
    return value ^ 0x5A;
}

int test_tst(unsigned value) {
    return (value & 0x5A) == 0;
}
