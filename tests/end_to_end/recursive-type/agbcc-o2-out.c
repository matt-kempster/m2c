? foo(? *, ? *);                                    /* static */

void test(void) {
    ? *sp4;

    sp4 = &sp4;
    foo(&subroutine_arg0);
    foo(&sp4, &sp4);
}
