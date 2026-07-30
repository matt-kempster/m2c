signed test(signed lhs, signed rhs) {
    return ((signed long long)lhs * rhs) >> 32;
}

unsigned test_unsigned(unsigned lhs, unsigned rhs) {
    return ((unsigned long long)lhs * rhs) >> 32;
}
