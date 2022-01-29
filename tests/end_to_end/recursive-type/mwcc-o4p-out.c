? foo(? *, ? *);                                    /* static */

void test(? *arg0, ? *arg1) {
    ? *sp8;                                         /* compiler-managed */
    ? *spC;
    ? *temp_r0;

    sp8 = arg0;
    temp_r0 = &spC;
    spC = arg1;
    sp8 = &sp8;
    spC = temp_r0;
    foo(&sp8, spC);
    sp8 = spC;
    foo((? *) sp8, spC);
}
