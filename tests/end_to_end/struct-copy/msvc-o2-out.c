void test(struct B *c, struct B *d) {
    M2C_MEMCPY(&a, &b, 0x64 * 4);
    M2C_MEMCPY(c, d, 0x19 * 4);
}
