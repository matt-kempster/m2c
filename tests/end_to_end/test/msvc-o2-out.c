? bar();                                            /* static */
extern s32 _foo;

void test(void) {
    bar();
    _foo = 4;
}
