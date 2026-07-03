? foo(? **, ? *);                                   /* static */

void test(? *arg0, ? *arg1) {
    arg0 = &arg0;
    arg1 = &arg1;
    foo((? **) &arg0, (? *) &arg1);
    arg0 = (? *) arg1;
    foo((? **) arg1, (? *) arg1);
}
