
volatile int global;

void test_s32(int a, int b, int c) {
    global = (a == b);
    global = (a != c);
    global = (a < b);
    global = (a <= b);
    global = (a > b);
    global = (a >= b);
    global = (a == 0);
    global = (b != 0);
    global = (a < 0);
    global = (a <= 0);
    global = (a > 0);
    global = (a >= 0);
}

void test_u32(unsigned int a, unsigned int b, unsigned int c) {
    global = (a == b);
    global = (a != c);
    global = (a < b);
    global = (a <= b);
    global = (a > b);
    global = (a >= b);
    global = (a == 0);
    global = (b != 0);
    global = (a < 0);
    global = (a <= 0);
    global = (a > 0);
    global = (a >= 0);
}

void test_s16(short a, short b, short c) {
    global = (a == b);
    global = (a != c);
    global = (a < b);
    global = (a <= b);
    global = (a > b);
    global = (a >= b);
    global = (a == 0);
    global = (b != 0);
    global = (a < 0);
    global = (a <= 0);
    global = (a > 0);
    global = (a >= 0);
}

void test_u16(unsigned short a, unsigned short b, unsigned short c) {
    global = (a == b);
    global = (a != c);
    global = (a < b);
    global = (a <= b);
    global = (a > b);
    global = (a >= b);
    global = (a == 0);
    global = (b != 0);
    global = (a < 0);
    global = (a <= 0);
    global = (a > 0);
    global = (a >= 0);
}

void test(void) {
    test_s32(1, 2, 3);
    test_u32(1, 2, 3);
    test_s16(1, 2, 3);
    test_u16(1, 2, 3);
}
