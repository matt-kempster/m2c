
volatile int global;

void test(unsigned int a, unsigned int b) {
    global = (a < b);
    global = (a <= b);
}
