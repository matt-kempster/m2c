unsigned test(unsigned lhs, unsigned rhs) {
    return lhs & rhs;
}

unsigned test_or(unsigned lhs, unsigned rhs) {
    return lhs | rhs;
}

unsigned test_xor(unsigned lhs, unsigned rhs) {
    return lhs ^ rhs;
}

unsigned test_not(unsigned value) {
    return ~value;
}

signed test_neg(signed value) {
    return -value;
}

int test_eq(int lhs, int rhs) {
    return lhs == rhs;
}

int test_ge(int lhs, int rhs) {
    return lhs >= rhs;
}

int test_gt(int lhs, int rhs) {
    return lhs > rhs;
}

int test_hs(unsigned lhs, unsigned rhs) {
    return lhs >= rhs;
}

int test_hi(unsigned lhs, unsigned rhs) {
    return lhs > rhs;
}

signed test_extsb(signed char value) {
    return value;
}

signed test_extsw(signed short value) {
    return value;
}

unsigned test_extub(unsigned char value) {
    return value;
}

unsigned test_extuw(unsigned short value) {
    return value;
}

unsigned test_swapw(unsigned value) {
    return (value << 16) | (value >> 16);
}
