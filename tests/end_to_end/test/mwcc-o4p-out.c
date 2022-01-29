? bar();                                            /* static */

void test(void) {
    bar();
    *NULL = 4;
}
