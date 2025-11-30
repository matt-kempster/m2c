? foo(? *);                                         /* static */

void test(? *arg4) {
    unksp0 = &unksp0;
    arg4 = &arg4;
    foo();
    unksp0 = arg4;
    foo((? *) arg4);
}
