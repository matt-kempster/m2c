? foo(? *, ? *);                                    /* static */

void test(? *arg0, ? *arg1) {
    ? *sp8;                                         /* compiler-managed */
    ? *spC;

    sp8 = arg0;
    spC = arg1;
    sp8 = &sp8;
    spC = &spC;
    foo(&sp8, spC);
    sp8 = spC;
    foo((? *) sp8, spC);
}
