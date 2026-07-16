unsigned test(unsigned value) {
    return value << 1;
}

signed test_shal(signed value) {
    return value << 1;
}

unsigned test_shlr(unsigned value) {
    return value >> 1;
}

signed test_shar(signed value) {
    return value >> 1;
}

unsigned test_shll2(unsigned value) {
    return value << 2;
}

unsigned test_shlr2(unsigned value) {
    return value >> 2;
}

unsigned test_shll8(unsigned value) {
    return value << 8;
}

unsigned test_shlr8(unsigned value) {
    return value >> 8;
}

unsigned test_shll16(unsigned value) {
    return value << 16;
}

unsigned test_shlr16(unsigned value) {
    return value >> 16;
}

unsigned test_rotl(unsigned value) {
    return (value << 1) | (value >> 31);
}

unsigned test_rotr(unsigned value) {
    return (value >> 1) | (value << 31);
}
