s32 bar(f32 x);                                     /* static */

void test(void) {
    *NULL = foo;
    *NULL = bar;
    *NULL = foo;
    *NULL = bar;
}
