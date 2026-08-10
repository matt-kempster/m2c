? foo(? *);                                         /* static */

void test(void) {
    ? *sp0;                                         /* compiler-managed */
    ? *sp4;

    sp0 = &sp0;
    sp4 = &sp4;
    foo();
    sp0 = &sp4;
    foo(&sp4);
}
