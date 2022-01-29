? bar();                                            /* static */
? foo();                                            /* static */

void test(void) {
    foo();
    *NULL = 1;
    bar();
}
