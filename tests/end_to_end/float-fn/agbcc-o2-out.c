? __extendsfdf2();                                  /* extern */
? __truncdfsf2();                                   /* extern */
? blah();                                           /* static */
? blahf();                                          /* static */

void test(void) {
    blahf();
    __extendsfdf2();
    blah();
    __truncdfsf2();
}
