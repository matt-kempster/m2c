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

unsigned test_shll3(unsigned value) {
    return value << 3;
}

unsigned test_shll10(unsigned value) {
    return value << 10;
}

unsigned test_shll18(unsigned value) {
    return value << 18;
}

signed test_shal3(signed value) {
    return value << 3;
}

signed test_shar2(signed value) {
    return value >> 2;
}

signed test_shar3(signed value) {
    return value >> 3;
}

signed test_mul3(signed value) {
    return value * 3;
}

signed test_mul5(signed value) {
    return value * 5;
}

signed test_mul10(signed value) {
    return value * 10;
}
